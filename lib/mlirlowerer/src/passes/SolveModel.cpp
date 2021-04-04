#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/ModelicaBuilder.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/SolveModel.h>
#include <modelica/mlirlowerer/passes/matching/Matching.h>
#include <modelica/mlirlowerer/passes/matching/SCCCollapsing.h>
#include <modelica/mlirlowerer/passes/matching/Schedule.h>
#include <modelica/mlirlowerer/passes/matching/SCCDependencyGraph.h>
#include <modelica/mlirlowerer/passes/matching/SVarDependencyGraph.h>
#include <modelica/mlirlowerer/passes/matching/VVarDependencyGraph.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/ModelBuilder.h>
#include <modelica/mlirlowerer/passes/model/SolveDer.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>
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

		// Create the loop
		auto loop = rewriter.create<ForOp>(location);

		{
			// Condition
			rewriter.setInsertionPointToStart(&loop.condition().front());
			mlir::Value currentTime = rewriter.create<LoadOp>(location, op.time());
			mlir::Value endTime = rewriter.create<ConstantOp>(location, op.endTime());
			mlir::Value condition = rewriter.create<LtOp>(location, BooleanType::get(op.getContext()), currentTime, endTime);
			rewriter.create<ConditionOp>(location, condition);
		}

		{
			// Body
			assert(op.body().getBlocks().size() == 1);
			rewriter.mergeBlocks(&op.body().front(), &loop.body().front());
		}

		{
			// Step
			rewriter.setInsertionPointToStart(&loop.step().front());
			mlir::Value currentTime = rewriter.create<LoadOp>(location, op.time());
			mlir::Value increasedTime = rewriter.create<AddOp>(location, currentTime.getType(), currentTime, timeStep);
			rewriter.create<StoreOp>(location, increasedTime, op.time());
			rewriter.create<YieldOp>(location);
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct EquationOpPattern : public mlir::OpRewritePattern<EquationOp>
{
	using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(EquationOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();

		// Create the assignment
		auto terminator = mlir::cast<EquationSidesOp>(op.body().front().getTerminator());
		rewriter.setInsertionPoint(terminator);

		for (auto [lhs, rhs] : llvm::zip(terminator.lhs(), terminator.rhs()))
		{
			if (auto loadOp = mlir::dyn_cast<LoadOp>(lhs.getDefiningOp()))
			{
				assert(loadOp.indexes().empty());
				rewriter.create<AssignmentOp>(location, rhs, loadOp.memory());
			}
			else
			{
				rewriter.create<AssignmentOp>(location, rhs, lhs);
			}
		}

		rewriter.eraseOp(terminator);

		// Inline the equation body
		rewriter.setInsertionPoint(op);
		rewriter.mergeBlockBefore(&op.body().front(), op);

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

			// Explicitate the equations so that the updated variable is the only
			// one on the left side of the equation.
			if (failed(explicitateEquations(model)))
				return signalPassFailure();

			// Calculate the values that the state variables will have in the next
			// iteration.
			if (failed(updateStates(model)))
				return signalPassFailure();

			module->dump();
		});

		// The model has been solved and we can now proceed to inline the
		// equations body and create the main simulation loop.

		mlir::ConversionTarget target(getContext());
		target.addLegalDialect<ModelicaDialect>();
		target.addIllegalOp<SimulationOp, EquationOp, ForEquationOp>();

		mlir::OwningRewritePatternList patterns;
		patterns.insert<SimulationOpPattern, EquationOpPattern>(&getContext());

		if (failed(applyPartialConversion(module, target, std::move(patterns))))
			return signalPassFailure();

		module->dump();
	}

	mlir::LogicalResult explicitateEquations(Model& model)
	{
		for (auto& equation : model.getEquations())
			if (auto res = equation->explicitate(); failed(res))
				return res;

		return mlir::success();
	}

	mlir::LogicalResult updateStates(Model& model)
	{
		mlir::OpBuilder builder(model.getOp());
		mlir::Location location = model.getOp()->getLoc();
		mlir::Value timeStep = builder.create<ConstantOp>(location, model.getOp().timeStep());

		builder.setInsertionPoint(model.getOp().body().back().getTerminator());

		for (auto& variable : model.getVariables())
		{
			if (!variable->isState())
				continue;

			mlir::Value var = builder.create<LoadOp>(location, variable->getReference());
			mlir::Value der = builder.create<LoadOp>(location, variable->getDer());
			mlir::Value newValue = builder.create<MulOp>(location, der.getType(), der, timeStep);
			newValue = builder.create<AddOp>(location, var.getType(), newValue, var);
			builder.create<StoreOp>(location, newValue, variable->getReference());
		}

		return mlir::success();
	}
};

std::unique_ptr<mlir::Pass> modelica::codegen::createSolveModelPass()
{
	return std::make_unique<SolveModelPass>();
}
