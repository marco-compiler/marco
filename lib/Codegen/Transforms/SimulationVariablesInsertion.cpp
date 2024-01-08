#include "marco/Codegen/Transforms/SimulationVariablesInsertion.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_SIMULATIONVARIABLESINSERTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class SimulationVariablesInsertionPass
      : public mlir::modelica::impl::SimulationVariablesInsertionPassBase<
            SimulationVariablesInsertionPass>
  {
    public:
      using SimulationVariablesInsertionPassBase<
          SimulationVariablesInsertionPass>
          ::SimulationVariablesInsertionPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult processModelOp(
          mlir::ModuleOp moduleOp, ModelOp modelOp);
  };
}

void SimulationVariablesInsertionPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  llvm::SmallVector<ModelOp> modelOps;

  for (ModelOp modelOp : moduleOp.getOps<ModelOp>()) {
    modelOps.push_back(modelOp);
  }

  for (ModelOp modelOp : modelOps) {
    if (mlir::failed(processModelOp(moduleOp, modelOp))) {
      return signalPassFailure();
    }
  }

  // Determine the analyses to be preserved.
  markAllAnalysesPreserved();
}

mlir::LogicalResult SimulationVariablesInsertionPass::processModelOp(
    mlir::ModuleOp moduleOp,
    ModelOp modelOp)
{
  mlir::OpBuilder builder(&getContext());
  builder.setInsertionPointToStart(moduleOp.getBody());

  for (VariableOp variable : modelOp.getVariables()) {
    builder.create<SimulationVariableOp>(
        variable.getLoc(), variable.getSymName(), variable.getVariableType());
  }

  return mlir::success();
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createSimulationVariablesInsertionPass()
  {
    return std::make_unique<SimulationVariablesInsertionPass>();
  }
}
