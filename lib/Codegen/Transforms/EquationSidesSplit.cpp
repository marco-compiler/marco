#include "marco/Codegen/Transforms/EquationSidesSplit.h"
#include "marco/Dialect/BaseModelica/ModelicaDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_EQUATIONSIDESSPLITPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class EquationSidesSplitPass
      : public impl::EquationSidesSplitPassBase<EquationSidesSplitPass>
  {
    public:
      using EquationSidesSplitPassBase<EquationSidesSplitPass>
          ::EquationSidesSplitPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult processModelOp(ModelOp modelOp);
  };
}

void EquationSidesSplitPass::runOnOperation()
{
  ModelOp modelOp = getOperation();

  if (mlir::failed(processModelOp(modelOp))) {
    return signalPassFailure();
  }
}

namespace
{
  class EquationSplitPattern : public mlir::OpRewritePattern<EquationOp>
  {
    public:
      using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          EquationOp op, mlir::PatternRewriter& rewriter) const override
      {
        auto equationSidesOp = mlir::cast<EquationSidesOp>(
            op.getBody()->getTerminator());

        auto lhsValues = equationSidesOp.getLhsValues();
        auto rhsValues = equationSidesOp.getRhsValues();

        if (lhsValues.size() != rhsValues.size()) {
          return mlir::failure();
        }

        size_t numOfElements = lhsValues.size();

        if (numOfElements <= 1) {
          return mlir::failure();
        }

        for (size_t i = 0; i < numOfElements; ++i) {
          mlir::IRMapping mapping;

          auto clonedEquation = mlir::cast<EquationOp>(
              rewriter.clone(*op.getOperation(), mapping));

          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToEnd(clonedEquation.getBody());

          auto clonedEquationSides = mlir::cast<EquationSidesOp>(
              clonedEquation.getBody()->getTerminator());

          auto clonedLhsOp = clonedEquationSides.getLhs().getDefiningOp();
          auto clonedRhsOp = clonedEquationSides.getRhs().getDefiningOp();

          mlir::Value lhs = mapping.lookup(lhsValues[i]);
          mlir::Value rhs = mapping.lookup(rhsValues[i]);

          auto newLhsOp = rewriter.create<EquationSideOp>(
              clonedLhsOp->getLoc(), lhs);

          auto newRhsOp = rewriter.create<EquationSideOp>(
              clonedRhsOp->getLoc(), rhs);

          rewriter.create<EquationSidesOp>(
              clonedEquationSides.getLoc(), newLhsOp, newRhsOp);

          rewriter.eraseOp(clonedEquationSides);
          rewriter.eraseOp(clonedLhsOp);
          rewriter.eraseOp(clonedRhsOp);
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };
}

mlir::LogicalResult EquationSidesSplitPass::processModelOp(ModelOp modelOp)
{
  mlir::ConversionTarget target(getContext());

  target.addDynamicallyLegalOp<EquationOp>([](EquationOp op) {
    auto equationSidesOp = mlir::cast<EquationSidesOp>(
        op.getBody()->getTerminator());

    auto lhsValues = equationSidesOp.getLhsValues();
    auto rhsValues = equationSidesOp.getRhsValues();

    return lhsValues.size() == 1 && rhsValues.size() == 1;
  });

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<EquationSplitPattern>(&getContext());
  return applyPartialConversion(modelOp, target, std::move(patterns));
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createEquationSidesSplitPass()
  {
    return std::make_unique<EquationSidesSplitPass>();
  }
}
