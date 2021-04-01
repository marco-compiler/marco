#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/SolveModel.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/ModelBuilder.h>
#include <modelica/mlirlowerer/passes/model/SolveDer.h>
#include <modelica/utils/Interval.hpp>
#include <variant>

using namespace modelica;
using namespace modelica::codegen;
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

/**
 * Model solver pass.
 * Its objective is to convert a descriptive model into an algorithmic one.
 */
class SolveModelPass: public mlir::PassWrapper<SolveModelPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<ModelicaDialect>();
	}

	void runOnOperation() override
	{
		auto module = getOperation();
		Model model;

		// Create the model
		module->walk([&](SimulationOp simulation) {
			ModelBuilder builder(model);

			simulation.walk([&](EquationOp equation) {
				builder.lower(equation);
			});

			simulation.walk([&](ForEquationOp forEquation) {
				builder.lower(forEquation);
			});
		});

		// Remove the derivative operations and allocate the appropriate buffers
		module.walk([&](SimulationOp simulation) {
			DerSolver solver(simulation, model);
			solver.solve();
		});



		module->dump();
	}
};

std::unique_ptr<mlir::Pass> modelica::createSolveModelPass()
{
	return std::make_unique<SolveModelPass>();
}
