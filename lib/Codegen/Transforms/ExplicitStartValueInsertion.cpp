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

        llvm::StringMap<StartOp> startOps;

        for (StartOp startOp : modelOp.getOps<StartOp>()) {
          startOps[startOp.getVariable()] = startOp;
        }

        for (VariableOp variableOp : modelOp.getOps<VariableOp>()) {
          VariableType variableType = variableOp.getVariableType();

          if (variableType.getElementType().isa<RecordType>()) {
            continue;
          }

          if (startOps.find(variableOp.getSymName()) == startOps.end()) {
            builder.setInsertionPointToEnd(modelOp.getBody());

            auto startOp = builder.create<StartOp>(
                variableOp.getLoc(), variableOp.getSymName(), false, false);

            assert(startOp.getBodyRegion().empty());
            mlir::Block* bodyBlock = builder.createBlock(&startOp.getBodyRegion());
            builder.setInsertionPointToStart(bodyBlock);

            mlir::Value zero = builder.create<ConstantOp>(
                variableOp.getLoc(), getZeroAttr(variableType.getElementType()));

            mlir::Value result = zero;

            if (!variableType.isScalar()) {
              result = builder.create<ArrayBroadcastOp>(
                  variableOp.getLoc(), variableType.toArrayType(), zero);
            }

            builder.create<YieldOp>(variableOp.getLoc(), result);
          }
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
