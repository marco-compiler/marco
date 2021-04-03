#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/ModelicaBuilder.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/SolveModel.h>
#include <modelica/mlirlowerer/passes/matching/Matching.h>
#include <modelica/mlirlowerer/passes/matching/SCCCollapsing.h>
#include <modelica/mlirlowerer/passes/matching/Schedule.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/ModelBuilder.h>
#include <modelica/mlirlowerer/passes/model/SolveDer.h>
#include <modelica/utils/Interval.hpp>
#include <variant>

using namespace modelica;
using namespace codegen;
using namespace model;

/*
static void scalarizeArrayEquations(SimulationOp simulation)
{
	simulation.walk([&](EquationOp equation) {
		auto& lhsRegion = equation.lhs();
		auto lhs = mlir::cast<YieldOp>(lhsRegion.front().getTerminator()).args();
		assert(lhs.size() == 1);

		if (auto pointerType = lhs[0].dyn_cast<PointerType>())
		{
			mlir::OpBuilder builder(equation);
			llvm::SmallVector<mlir::Value, 3> inductions;

			for (size_t i = 0; i < pointerType.getRank(); ++i)
				inductions.push_back(builder.create<InductionOp>(equation.getLoc(), 0, pointerType.getShape()[i]));

			auto forLoop = builder.create<ForEquationOp>(equation.getLoc(), inductions);

			mlir::BlockAndValueMapping map;
			equation.lhs().cloneInto(&forLoop.lhs(), forLoop.lhs().begin(), map);

			equation->erase();
		}
	});
}
 */

struct SimulationOpPattern : public mlir::OpRewritePattern<SimulationOp>
{
	using mlir::OpRewritePattern<SimulationOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(SimulationOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();

		// Initiate the time variable
		mlir::Value startTime = rewriter.create<ConstantOp>(location, op.startTime());
		rewriter.create<StoreOp>(location, startTime, op.time());
		mlir::Value timeStep = rewriter.create<ConstantOp>(location, op.timeStep());

		// Split the current block
		mlir::Block* currentBlock = rewriter.getInsertionBlock(); // initBlock
		mlir::Block* conditionBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
		mlir::Block* continuationBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

		{
			// Inline the body and increment the time at its end
			mlir::Block* body = &op.body().front();
			rewriter.inlineRegionBefore(op.body(), continuationBlock);
			rewriter.setInsertionPointToEnd(body);
			mlir::Value currentTime = rewriter.create<LoadOp>(location, op.time());
			mlir::Value increasedTime = rewriter.create<AddOp>(location, currentTime.getType(), currentTime, timeStep);
			rewriter.create<StoreOp>(location, increasedTime, op.time());
			rewriter.create<mlir::BranchOp>(location, conditionBlock);
		}

		// Start the for loop by branching to the "condition" region
		rewriter.setInsertionPointToEnd(currentBlock);
		rewriter.create<mlir::BranchOp>(location, conditionBlock);

		{
			// Check the condition
			rewriter.setInsertionPointToStart(conditionBlock);
			mlir::Value currentTime = rewriter.create<LoadOp>(location, op.time());
		}

		// Create the model
		Model model(op, {}, {});
		ModelBuilder builder(model);

		op.walk([&](EquationOp equation) {
			builder.lower(equation);
		});

		op.walk([&](ForEquationOp forEquation) {
			builder.lower(forEquation);
		});

		// Remove the derivative operations and allocate the appropriate buffers
		DerSolver solver(op, model);
		solver.solve();

		// Match
		if (auto res = modelica::codegen::model::match(model, 1000); failed(res))
			return res;

		// Solve SCC
		if (auto res = solveSCC(model, 1000); failed(res))
			return res;

		// Schedule
		if (auto res = schedule(model); failed(res))
			return res;

		// Create the sequential code
		for (auto& equation : model.getEquations())
		{
			if (auto equationOp = mlir::dyn_cast<EquationOp>(equation->getOp()))
			{
				//rewriter.eraseOp(equationOp.lhs().front().getTerminator());
				//rewriter.eraseOp(equationOp.rhs().front().getTerminator());
				//rewriter.mergeBlockBefore(&equationOp.lhs().front(), equationOp);
				//rewriter.mergeBlockBefore(&equationOp.rhs().front(), equationOp);
			}

			rewriter.eraseOp(equation->getOp());
		}



		op->getParentOp()->dump();

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

/**
 * Model solver pass.
 * Its objective is to convert a descriptive (and thus not sequential) model
 * into an algorithmic one.
 */
class SolveModelPass: public mlir::PassWrapper<SolveModelPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<mlir::StandardOpsDialect>();
		registry.insert<ModelicaDialect>();
	}

	void runOnOperation() override
	{
		auto module = getOperation();

		/*
		mlir::ConversionTarget target(getContext());
		target.addLegalDialect<ModelicaDialect>();
		target.addIllegalOp<SimulationOp>();

		mlir::OwningRewritePatternList patterns;
		patterns.insert<SimulationOpPattern>(&getContext());

		if (failed(applyPartialConversion(module, target, std::move(patterns))))
			return signalPassFailure();
		 */

		module->walk([&](SimulationOp simulation) {
			// Create the model
			Model model(simulation, {}, {});
			ModelBuilder builder(model);

			simulation.walk([&](EquationOp equation) {
				builder.lower(equation);
			});

			simulation.walk([&](ForEquationOp forEquation) {
				builder.lower(forEquation);
			});

			// Remove the derivative operations and allocate the appropriate buffers
			DerSolver solver(simulation, model);
			solver.solve();

			// Match
			if (failed(match(model, 1000)))
				return signalPassFailure();

			// Solve SCC
			if (failed(solveSCC(model, 1000)))
				return signalPassFailure();

			// Schedule
			if (failed(schedule(model)))
				return signalPassFailure();

			module->dump();

			// Create the sequential code
			if (failed(addApproximation(model)))
				return signalPassFailure();

			module->dump();
		});
	}

	mlir::LogicalResult addApproximation(Model& model)
	{
		mlir::OpBuilder builder(model.getOp());
		mlir::Location location = model.getOp()->getLoc();
		mlir::Value timeStep = builder.create<ConstantOp>(location, model.getOp().timeStep());

		for (auto& equation : model.getEquations())
			if (auto res = equation->explicitate(); failed(res))
				return res;

		return mlir::success();
	}
};

std::unique_ptr<mlir::Pass> modelica::codegen::createSolveModelPass()
{
	return std::make_unique<SolveModelPass>();
}
