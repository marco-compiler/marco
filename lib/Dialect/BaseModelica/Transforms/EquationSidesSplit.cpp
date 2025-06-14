#include "marco/Dialect/BaseModelica/Transforms/EquationSidesSplit.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_EQUATIONSIDESSPLITPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class EquationSidesSplitPass
    : public impl::EquationSidesSplitPassBase<EquationSidesSplitPass> {
public:
  using EquationSidesSplitPassBase<
      EquationSidesSplitPass>::EquationSidesSplitPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult processModelOp(ModelOp modelOp);

  mlir::LogicalResult splitEquation(mlir::RewriterBase &rewriter,
                                    EquationOp equationOp);
};
} // namespace

void EquationSidesSplitPass::runOnOperation() {
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

mlir::LogicalResult EquationSidesSplitPass::processModelOp(ModelOp modelOp) {
  llvm::SmallVector<EquationOp> equationOps;

  modelOp.walk([&](EquationOp equationOp) {
    auto equationSidesOp =
        mlir::cast<EquationSidesOp>(equationOp.getBody()->getTerminator());

    auto lhsValues = equationSidesOp.getLhsValues();
    auto rhsValues = equationSidesOp.getRhsValues();

    if (lhsValues.size() != 1 && lhsValues.size() == rhsValues.size()) {
      equationOps.push_back(equationOp);
    }
  });

  mlir::IRRewriter rewriter(modelOp);

  for (EquationOp equationOp : equationOps) {
    if (mlir::failed(splitEquation(rewriter, equationOp))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult
EquationSidesSplitPass::splitEquation(mlir::RewriterBase &rewriter,
                                      EquationOp op) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  auto equationSidesOp =
      mlir::cast<EquationSidesOp>(op.getBody()->getTerminator());

  auto lhsValues = equationSidesOp.getLhsValues();
  auto rhsValues = equationSidesOp.getRhsValues();

  if (lhsValues.size() != rhsValues.size()) {
    return mlir::failure();
  }

  size_t numOfElements = lhsValues.size();

  for (size_t i = 0; i < numOfElements; ++i) {
    mlir::IRMapping mapping;
    rewriter.setInsertionPointAfter(op);

    auto clonedEquation =
        mlir::cast<EquationOp>(rewriter.clone(*op.getOperation(), mapping));

    rewriter.setInsertionPointToEnd(clonedEquation.getBody());

    auto clonedEquationSides =
        mlir::cast<EquationSidesOp>(clonedEquation.getBody()->getTerminator());

    auto clonedLhsOp = clonedEquationSides.getLhs().getDefiningOp();
    auto clonedRhsOp = clonedEquationSides.getRhs().getDefiningOp();

    mlir::Value lhs = mapping.lookup(lhsValues[i]);
    mlir::Value rhs = mapping.lookup(rhsValues[i]);

    auto newLhsOp = rewriter.create<EquationSideOp>(clonedLhsOp->getLoc(), lhs);
    auto newRhsOp = rewriter.create<EquationSideOp>(clonedRhsOp->getLoc(), rhs);

    rewriter.create<EquationSidesOp>(clonedEquationSides.getLoc(), newLhsOp,
                                     newRhsOp);

    rewriter.eraseOp(clonedEquationSides);
    rewriter.eraseOp(clonedLhsOp);
    rewriter.eraseOp(clonedRhsOp);
  }

  rewriter.eraseOp(op);
  return mlir::success();
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createEquationSidesSplitPass() {
  return std::make_unique<EquationSidesSplitPass>();
}
} // namespace mlir::bmodelica
