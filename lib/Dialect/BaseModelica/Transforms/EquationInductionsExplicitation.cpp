#include "marco/Dialect/BaseModelica/Transforms/EquationInductionsExplicitation.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_EQUATIONINDUCTIONSEXPLICITATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class EquationInductionsExplicitationPass
    : public impl::EquationInductionsExplicitationPassBase<
          EquationInductionsExplicitationPass> {
public:
  using EquationInductionsExplicitationPassBase<
      EquationInductionsExplicitationPass>::
      EquationInductionsExplicitationPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult processModelOp(ModelOp modelOp);
};
} // namespace

void EquationInductionsExplicitationPass::runOnOperation() {
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), modelOps, [&](mlir::Operation *op) {
            return processModelOp(mlir::cast<ModelOp>(op));
          }))) {
    return signalPassFailure();
  }
}

namespace {
class EquationImplicitInductionsPattern
    : public mlir::OpRewritePattern<EquationOp> {
public:
  using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EquationOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto equationSidesOp =
        mlir::cast<EquationSidesOp>(op.getBody()->getTerminator());

    mlir::Value lhsValue = equationSidesOp.getLhsValues()[0];
    mlir::Value rhsValue = equationSidesOp.getRhsValues()[0];

    auto lhsTensorType = mlir::cast<mlir::TensorType>(lhsValue.getType());
    auto rhsTensorType = mlir::cast<mlir::TensorType>(rhsValue.getType());

    llvm::SmallVector<mlir::Value> inductions;

    for (size_t i = 0, e = lhsTensorType.getRank(); i < e; ++i) {
      if (lhsTensorType.isDynamicDim(i)) {
        assert(!rhsTensorType.isDynamicDim(i));

        auto forOp = rewriter.create<ForEquationOp>(
            op.getLoc(), 0, rhsTensorType.getDimSize(i) - 1, 1);

        rewriter.setInsertionPointToStart(forOp.bodyBlock());
        inductions.push_back(forOp.induction());
      } else {
        assert(!lhsTensorType.isDynamicDim(i));

        auto forOp = rewriter.create<ForEquationOp>(
            op.getLoc(), 0, lhsTensorType.getDimSize(i) - 1, 1);

        rewriter.setInsertionPointToStart(forOp.bodyBlock());
        inductions.push_back(forOp.induction());
      }
    }

    mlir::IRMapping mapping;

    auto clonedEquation =
        mlir::cast<EquationOp>(rewriter.clone(*op.getOperation(), mapping));

    auto clonedEquationSidesOp =
        mlir::cast<EquationSidesOp>(clonedEquation.getBody()->getTerminator());

    auto clonedLhsOp = clonedEquationSidesOp.getLhs().getDefiningOp();
    auto clonedRhsOp = clonedEquationSidesOp.getRhs().getDefiningOp();

    rewriter.setInsertionPointAfter(clonedEquationSidesOp);

    assert(clonedEquationSidesOp.getLhsValues().size() == 1);
    assert(clonedEquationSidesOp.getRhsValues().size() == 1);

    auto newLhs = rewriter.create<TensorExtractOp>(
        op.getLoc(), clonedEquationSidesOp.getLhsValues()[0], inductions);

    auto newRhs = rewriter.create<TensorExtractOp>(
        op.getLoc(), clonedEquationSidesOp.getRhsValues()[0], inductions);

    auto newLhsOp = rewriter.create<EquationSideOp>(clonedLhsOp->getLoc(),
                                                    newLhs->getResults());

    auto newRhsOp = rewriter.create<EquationSideOp>(clonedRhsOp->getLoc(),
                                                    newRhs->getResults());

    rewriter.create<EquationSidesOp>(clonedEquationSidesOp.getLoc(), newLhsOp,
                                     newRhsOp);

    rewriter.eraseOp(clonedEquationSidesOp);
    rewriter.eraseOp(clonedLhsOp);
    rewriter.eraseOp(clonedRhsOp);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};
} // namespace

mlir::LogicalResult
EquationInductionsExplicitationPass::processModelOp(ModelOp modelOp) {
  mlir::ConversionTarget target(getContext());

  target.addDynamicallyLegalOp<EquationOp>([](EquationOp op) {
    auto equationSidesOp =
        mlir::cast<EquationSidesOp>(op.getBody()->getTerminator());

    auto lhsValues = equationSidesOp.getLhsValues();
    auto rhsValues = equationSidesOp.getRhsValues();

    if (lhsValues.size() != 1 || rhsValues.size() != 1) {
      return true;
    }

    auto lhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(lhsValues[0].getType());
    auto rhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(rhsValues[0].getType());

    if (!lhsTensorType || !rhsTensorType) {
      return true;
    }

    if (lhsTensorType.getRank() != rhsTensorType.getRank()) {
      return true;
    }

    for (size_t i = 0, e = lhsTensorType.getRank(); i < e; ++i) {
      if (lhsTensorType.isDynamicDim(i) && rhsTensorType.isDynamicDim(i)) {
        return true;
      }

      if (lhsTensorType.isDynamicDim(i) || rhsTensorType.isDynamicDim(i)) {
        continue;
      }

      if (lhsTensorType.getDimSize(i) != rhsTensorType.getDimSize(i)) {
        return true;
      }
    }

    return false;
  });

  target.markUnknownOpDynamicallyLegal(
      [](mlir::Operation *op) { return true; });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<EquationImplicitInductionsPattern>(&getContext());
  return applyPartialConversion(modelOp, target, std::move(patterns));
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createEquationInductionsExplicitationPass() {
  return std::make_unique<EquationInductionsExplicitationPass>();
}
} // namespace mlir::bmodelica
