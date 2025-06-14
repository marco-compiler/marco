#include "marco/Dialect/BaseModelica/Transforms/EquationInductionsExplicitation.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

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

  mlir::LogicalResult explicitateInductions(mlir::RewriterBase &rewriter,
                                            EquationOp equationOp);
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

mlir::LogicalResult
EquationInductionsExplicitationPass::processModelOp(ModelOp modelOp) {
  llvm::SmallVector<EquationOp> equationOps;

  modelOp.walk([&](EquationOp equationOp) {
    auto equationSidesOp =
        mlir::cast<EquationSidesOp>(equationOp.getBody()->getTerminator());

    auto lhsValues = equationSidesOp.getLhsValues();
    auto rhsValues = equationSidesOp.getRhsValues();

    if (lhsValues.size() != 1 || rhsValues.size() != 1) {
      // Can't add induction variables in case of more elements on one of the
      // equation sides. This is a safety check, as the equation sides should
      // have been already split using the dedicated pass.
      return;
    }

    auto lhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(lhsValues[0].getType());

    auto rhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(rhsValues[0].getType());

    if (!lhsTensorType || !rhsTensorType) {
      return;
    }

    if (lhsTensorType.getRank() != rhsTensorType.getRank()) {
      return;
    }

    for (size_t i = 0, e = lhsTensorType.getRank(); i < e; ++i) {
      if (lhsTensorType.isDynamicDim(i) && rhsTensorType.isDynamicDim(i)) {
        return;
      }

      if (lhsTensorType.isDynamicDim(i) || rhsTensorType.isDynamicDim(i)) {
        continue;
      }

      if (lhsTensorType.getDimSize(i) != rhsTensorType.getDimSize(i)) {
        return;
      }
    }

    equationOps.push_back(equationOp);
  });

  mlir::IRRewriter rewriter(modelOp);

  for (EquationOp equationOp : equationOps) {
    if (mlir::failed(explicitateInductions(rewriter, equationOp))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult EquationInductionsExplicitationPass::explicitateInductions(
    mlir::RewriterBase &rewriter, EquationOp equationOp) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(equationOp);

  auto equationSidesOp =
      mlir::cast<EquationSidesOp>(equationOp.getBody()->getTerminator());

  mlir::Value lhsValue = equationSidesOp.getLhsValues()[0];
  mlir::Value rhsValue = equationSidesOp.getRhsValues()[0];

  auto lhsTensorType = mlir::cast<mlir::TensorType>(lhsValue.getType());
  auto rhsTensorType = mlir::cast<mlir::TensorType>(rhsValue.getType());

  llvm::SmallVector<mlir::Value> inductions;

  for (size_t i = 0, e = lhsTensorType.getRank(); i < e; ++i) {
    if (lhsTensorType.isDynamicDim(i)) {
      assert(!rhsTensorType.isDynamicDim(i));

      auto forOp = rewriter.create<ForEquationOp>(
          equationOp.getLoc(), 0, rhsTensorType.getDimSize(i) - 1, 1);

      rewriter.setInsertionPointToStart(forOp.bodyBlock());
      inductions.push_back(forOp.induction());
    } else {
      assert(!lhsTensorType.isDynamicDim(i));

      auto forOp = rewriter.create<ForEquationOp>(
          equationOp.getLoc(), 0, lhsTensorType.getDimSize(i) - 1, 1);

      rewriter.setInsertionPointToStart(forOp.bodyBlock());
      inductions.push_back(forOp.induction());
    }
  }

  mlir::IRMapping mapping;

  auto clonedEquation = mlir::cast<EquationOp>(
      rewriter.clone(*equationOp.getOperation(), mapping));

  auto clonedEquationSidesOp =
      mlir::cast<EquationSidesOp>(clonedEquation.getBody()->getTerminator());

  auto clonedLhsOp = clonedEquationSidesOp.getLhs().getDefiningOp();
  auto clonedRhsOp = clonedEquationSidesOp.getRhs().getDefiningOp();

  rewriter.setInsertionPointAfter(clonedEquationSidesOp);

  assert(clonedEquationSidesOp.getLhsValues().size() == 1);
  assert(clonedEquationSidesOp.getRhsValues().size() == 1);

  auto newLhs = rewriter.create<TensorExtractOp>(
      equationOp.getLoc(), clonedEquationSidesOp.getLhsValues()[0], inductions);

  auto newRhs = rewriter.create<TensorExtractOp>(
      equationOp.getLoc(), clonedEquationSidesOp.getRhsValues()[0], inductions);

  auto newLhsOp = rewriter.create<EquationSideOp>(clonedLhsOp->getLoc(),
                                                  newLhs->getResults());

  auto newRhsOp = rewriter.create<EquationSideOp>(clonedRhsOp->getLoc(),
                                                  newRhs->getResults());

  rewriter.create<EquationSidesOp>(clonedEquationSidesOp.getLoc(), newLhsOp,
                                   newRhsOp);

  rewriter.eraseOp(clonedEquationSidesOp);
  rewriter.eraseOp(clonedLhsOp);
  rewriter.eraseOp(clonedRhsOp);

  rewriter.eraseOp(equationOp);
  return mlir::success();
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createEquationInductionsExplicitationPass() {
  return std::make_unique<EquationInductionsExplicitationPass>();
}
} // namespace mlir::bmodelica
