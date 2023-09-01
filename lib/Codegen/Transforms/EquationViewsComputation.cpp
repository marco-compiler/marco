#include "marco/Codegen/Transforms/EquationViewsComputation.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_EQUATIONVIEWSCOMPUTATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class EquationViewsComputationPass
      : public impl::EquationViewsComputationPassBase<
          EquationViewsComputationPass>
  {
    public:
      using EquationViewsComputationPassBase::EquationViewsComputationPassBase;

      void runOnOperation() override;

    private:
      void createViews(
          mlir::OpBuilder& builder,
          EquationInstanceOp equationOp);
  };
}

void EquationViewsComputationPass::runOnOperation()
{
  ModelOp modelOp = getOperation();
  mlir::OpBuilder builder(modelOp);

  llvm::SmallVector<EquationInstanceOp> equationOps;

  for (EquationInstanceOp op : modelOp.getOps<EquationInstanceOp>()) {
    if (!op.getViewElementIndex()) {
      equationOps.push_back(op);
    }
  }

  for (EquationInstanceOp op : equationOps) {
    createViews(builder, op);
    op.erase();
  }
}

void EquationViewsComputationPass::createViews(
    mlir::OpBuilder& builder,
    EquationInstanceOp equationOp)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(equationOp);

  EquationTemplateOp equationTemplateOp = equationOp.getTemplate();

  mlir::Block* bodyBlock = equationTemplateOp.getBody();

  auto equationSidesOp =
      mlir::cast<EquationSidesOp>(bodyBlock->getTerminator());

  assert(equationSidesOp.getLhsValues().size() ==
         equationSidesOp.getRhsValues().size());

  for (size_t i = 0, e = equationSidesOp.getLhsValues().size(); i < e; ++i) {
    auto clonedOp = builder.cloneWithoutRegions(equationOp);
    clonedOp.setViewElementIndex(i);
  }
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createEquationViewsComputationPass()
  {
    return std::make_unique<EquationViewsComputationPass>();
  }
}
