#include "marco/Codegen/Transforms/ExplicitStartValueInsertion.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_EXPLICITSTARTVALUEINSERTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class ExplicitStartValueInsertionPass
      : public impl::ExplicitStartValueInsertionPassBase<
          ExplicitStartValueInsertionPass>
  {
    public:
      using ExplicitStartValueInsertionPassBase
        ::ExplicitStartValueInsertionPassBase;

      void runOnOperation() override
      {
        ModelOp modelOp = getOperation();
        mlir::OpBuilder builder(modelOp);

        // Collect the variables.
        llvm::SmallVector<VariableOp> variableOps;
        modelOp.collectVariables(variableOps);

        // Collect the existing 'start' operations.
        llvm::StringMap<StartOp> startOps;

        for (StartOp startOp : modelOp.getOps<StartOp>()) {
          startOps[startOp.getVariable()] = startOp;
        }

        // Create the missing 'start' operations.
        for (VariableOp variableOp : variableOps) {
          VariableType variableType = variableOp.getVariableType();

          if (startOps.contains(variableOp.getSymName())) {
            // The variable already has a start value.
            continue;
          }

          auto zeroMaterializableType =
              variableType.getElementType().dyn_cast<ZeroMaterializableType>();

          if (!zeroMaterializableType) {
            // Proceed only if a zero-valued constant can be materialized.
            continue;
          }

          // Append the new operation at the end of the model.
          builder.setInsertionPointToEnd(modelOp.getBody());

          auto startOp = builder.create<StartOp>(
              variableOp.getLoc(), variableOp.getSymName(), false, false);

          assert(startOp.getBodyRegion().empty());

          mlir::Block* bodyBlock =
              builder.createBlock(&startOp.getBodyRegion());

          builder.setInsertionPointToStart(bodyBlock);

          mlir::Value zero =
              zeroMaterializableType.materializeZeroValuedConstant(
                  builder, variableOp.getLoc());

          mlir::Value result = zero;

          if (!variableType.isScalar()) {
            result = builder.create<ArrayBroadcastOp>(
                variableOp.getLoc(), variableType.toArrayType(), zero);
          }

          builder.create<YieldOp>(variableOp.getLoc(), result);
        }
      }
  };
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createExplicitStartValueInsertionPass()
  {
    return std::make_unique<ExplicitStartValueInsertionPass>();
  }
}
