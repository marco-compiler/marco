#include "marco/Dialect/BaseModelica/Transforms/ExplicitStartValueInsertion.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_EXPLICITSTARTVALUEINSERTIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

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
          assert(startOp.getVariable().getNestedReferences().empty());
          startOps[startOp.getVariable().getRootReference()] = startOp;
        }

        // Create the missing 'start' operations.
        for (VariableOp variableOp : variableOps) {
          VariableType variableType = variableOp.getVariableType();

          if (startOps.contains(variableOp.getSymName())) {
            // The variable already has a start value.
            continue;
          }

          auto constantMaterializableType =
              variableType.getElementType()
                  .dyn_cast<ConstantMaterializableTypeInterface>();

          if (!constantMaterializableType) {
            // Proceed only if a zero-valued constant can be materialized.
            continue;
          }

          // Append the new operation at the end of the model.
          builder.setInsertionPointToEnd(modelOp.getBody());

          auto startOp = builder.create<StartOp>(
              variableOp.getLoc(),
              mlir::SymbolRefAttr::get(variableOp.getSymNameAttr()),
              false, false, true);

          assert(startOp.getBodyRegion().empty());

          mlir::Block* bodyBlock =
              builder.createBlock(&startOp.getBodyRegion());

          builder.setInsertionPointToStart(bodyBlock);

          mlir::Value zero = constantMaterializableType.materializeIntConstant(
              builder, variableOp.getLoc(), 0);

          mlir::Value result = zero;

          if (!variableType.isScalar()) {
            result = builder.create<TensorBroadcastOp>(
                variableOp.getLoc(), variableType.toTensorType(), zero);
          }

          builder.create<YieldOp>(variableOp.getLoc(), result);
        }
      }
  };
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createExplicitStartValueInsertionPass()
  {
    return std::make_unique<ExplicitStartValueInsertionPass>();
  }
}
