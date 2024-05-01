#include "marco/Codegen/Transforms/EquationInductionsExplicitation.h"
#include "marco/Dialect/BaseModelica/ModelicaDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_EQUATIONINDUCTIONSEXPLICITATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class EquationInductionsExplicitationPass
      : public impl::EquationInductionsExplicitationPassBase<
            EquationInductionsExplicitationPass>
  {
    public:
      using EquationInductionsExplicitationPassBase<
          EquationInductionsExplicitationPass>
          ::EquationInductionsExplicitationPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult processModelOp(ModelOp modelOp);
  };
}

void EquationInductionsExplicitationPass::runOnOperation()
{
  ModelOp modelOp = getOperation();

  if (mlir::failed(processModelOp(modelOp))) {
    return signalPassFailure();
  }
}

namespace
{
  class EquationImplicitInductionsPattern
      : public mlir::OpRewritePattern<EquationOp>
  {
    public:
      using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          EquationOp op, mlir::PatternRewriter& rewriter) const override
      {
        auto equationSidesOp = mlir::cast<EquationSidesOp>(
            op.getBody()->getTerminator());

        mlir::Value lhsValue = equationSidesOp.getLhsValues()[0];
        mlir::Value rhsValue = equationSidesOp.getRhsValues()[0];

        auto lhsArrayType = lhsValue.getType().cast<ArrayType>();
        auto rhsArrayType = rhsValue.getType().cast<ArrayType>();

        llvm::SmallVector<mlir::Value> inductions;

        for (size_t i = 0, e = lhsArrayType.getRank(); i < e; ++i) {
          if (lhsArrayType.isDynamicDim(i)) {
            assert(!rhsArrayType.isDynamicDim(i));

            auto forOp = rewriter.create<ForEquationOp>(
                op.getLoc(), 0, rhsArrayType.getDimSize(i) - 1, 1);

            rewriter.setInsertionPointToStart(forOp.bodyBlock());
            inductions.push_back(forOp.induction());
          } else {
            assert(!lhsArrayType.isDynamicDim(i));

            auto forOp = rewriter.create<ForEquationOp>(
                op.getLoc(), 0, lhsArrayType.getDimSize(i) - 1, 1);

            rewriter.setInsertionPointToStart(forOp.bodyBlock());
            inductions.push_back(forOp.induction());
          }
        }

        mlir::IRMapping mapping;

        auto clonedEquation = mlir::cast<EquationOp>(
            rewriter.clone(*op.getOperation(), mapping));

        auto clonedEquationSidesOp = mlir::cast<EquationSidesOp>(
            clonedEquation.getBody()->getTerminator());

        auto clonedLhsOp = clonedEquationSidesOp.getLhs().getDefiningOp();
        auto clonedRhsOp = clonedEquationSidesOp.getRhs().getDefiningOp();

        rewriter.setInsertionPointAfter(clonedEquationSidesOp);

        assert(clonedEquationSidesOp.getLhsValues().size() == 1);
        assert(clonedEquationSidesOp.getRhsValues().size() == 1);

        auto newLhs = rewriter.create<LoadOp>(
            op.getLoc(), clonedEquationSidesOp.getLhsValues()[0], inductions);

        auto newRhs = rewriter.create<LoadOp>(
            op.getLoc(), clonedEquationSidesOp.getRhsValues()[0], inductions);

        auto newLhsOp = rewriter.create<EquationSideOp>(
            clonedLhsOp->getLoc(), newLhs->getResults());

        auto newRhsOp = rewriter.create<EquationSideOp>(
            clonedRhsOp->getLoc(), newRhs->getResults());

        rewriter.create<EquationSidesOp>(
            clonedEquationSidesOp.getLoc(), newLhsOp, newRhsOp);

        rewriter.eraseOp(clonedEquationSidesOp);
        rewriter.eraseOp(clonedLhsOp);
        rewriter.eraseOp(clonedRhsOp);

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };
}

mlir::LogicalResult
EquationInductionsExplicitationPass::processModelOp(ModelOp modelOp)
{
  mlir::ConversionTarget target(getContext());

  target.addDynamicallyLegalOp<EquationOp>([](EquationOp op) {
    auto equationSidesOp = mlir::cast<EquationSidesOp>(
        op.getBody()->getTerminator());

    auto lhsValues = equationSidesOp.getLhsValues();
    auto rhsValues = equationSidesOp.getRhsValues();

    if (lhsValues.size() != 1 || rhsValues.size() != 1) {
      return true;
    }

    auto lhsArrayType = lhsValues[0].getType().dyn_cast<ArrayType>();
    auto rhsArrayType = rhsValues[0].getType().dyn_cast<ArrayType>();

    if (!lhsArrayType || !rhsArrayType) {
      return true;
    }

    if (lhsArrayType.getRank() != rhsArrayType.getRank()) {
      return true;
    }

    for (size_t i = 0, e = lhsArrayType.getRank(); i < e; ++i) {
      if (lhsArrayType.isDynamicDim(i) && rhsArrayType.isDynamicDim(i)) {
        return true;
      }

      if (lhsArrayType.isDynamicDim(i) || rhsArrayType.isDynamicDim(i)) {
        continue;
      }

      if (lhsArrayType.getDimSize(i) != rhsArrayType.getDimSize(i)) {
        return true;
      }
    }

    return false;
  });

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<EquationImplicitInductionsPattern>(&getContext());
  return applyPartialConversion(modelOp, target, std::move(patterns));
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createEquationInductionsExplicitationPass()
  {
    return std::make_unique<EquationInductionsExplicitationPass>();
  }
}
