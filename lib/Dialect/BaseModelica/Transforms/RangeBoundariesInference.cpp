#include "marco/Dialect/BaseModelica/Transforms/RangeBoundariesInference.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_RANGEBOUNDARIESINFERENCEPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class TensorViewOpPattern : public mlir::OpRewritePattern<TensorViewOp> {
public:
  using mlir::OpRewritePattern<TensorViewOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TensorViewOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (llvm::all_of(op.getSubscriptions(), [](mlir::Value index) {
          mlir::Operation *definingOp = index.getDefiningOp();

          if (!definingOp) {
            return true;
          }

          return !mlir::isa<UnboundedRangeOp>(definingOp);
        })) {
      return mlir::failure();
    }

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::Value> newIndices;

    for (size_t i = 0, e = op.getSubscriptions().size(); i < e; ++i) {
      mlir::Value index = op.getSubscriptions()[i];

      if (index.getDefiningOp<UnboundedRangeOp>()) {
        mlir::Value lowerBound =
            rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));

        mlir::Value dimensionIndex = rewriter.create<ConstantOp>(
            loc, rewriter.getIndexAttr(static_cast<int64_t>(i)));

        mlir::Value dimensionSize = rewriter.create<mlir::tensor::DimOp>(
            loc, op.getSource(), dimensionIndex);

        mlir::Value offset =
            rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(-1));

        mlir::Value upperBound = rewriter.create<AddOp>(
            loc, rewriter.getIndexType(), dimensionSize, offset);

        mlir::Value step =
            rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));

        mlir::Value range = rewriter.create<RangeOp>(
            loc, RangeType::get(getContext(), rewriter.getIndexType()),
            lowerBound, upperBound, step);

        newIndices.push_back(range);
        continue;
      }

      newIndices.push_back(index);
    }

    rewriter.replaceOpWithNewOp<TensorViewOp>(op, op.getSource(), newIndices);

    return mlir::success();
  }
};

class TensorInsertSliceOpPattern
    : public mlir::OpRewritePattern<TensorInsertSliceOp> {
public:
  using mlir::OpRewritePattern<TensorInsertSliceOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TensorInsertSliceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (llvm::all_of(op.getSubscriptions(), [](mlir::Value index) {
          mlir::Operation *definingOp = index.getDefiningOp();

          if (!definingOp) {
            return true;
          }

          return !mlir::isa<UnboundedRangeOp>(definingOp);
        })) {
      return mlir::failure();
    }

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::Value> newIndices;

    for (size_t i = 0, e = op.getSubscriptions().size(); i < e; ++i) {
      mlir::Value index = op.getSubscriptions()[i];

      if (index.getDefiningOp<UnboundedRangeOp>()) {
        mlir::Value lowerBound =
            rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));

        mlir::Value dimensionIndex = rewriter.create<ConstantOp>(
            loc, rewriter.getIndexAttr(static_cast<int64_t>(i)));

        mlir::Value dimensionSize = rewriter.create<mlir::tensor::DimOp>(
            loc, op.getDestination(), dimensionIndex);

        mlir::Value offset =
            rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(-1));

        mlir::Value upperBound = rewriter.create<AddOp>(
            loc, rewriter.getIndexType(), dimensionSize, offset);

        mlir::Value step =
            rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));

        mlir::Value range = rewriter.create<RangeOp>(
            loc, RangeType::get(getContext(), rewriter.getIndexType()),
            lowerBound, upperBound, step);

        newIndices.push_back(range);
        continue;
      }

      newIndices.push_back(index);
    }

    rewriter.replaceOpWithNewOp<TensorInsertSliceOp>(
        op, op.getValue(), op.getDestination(), newIndices);

    return mlir::success();
  }
};
} // namespace

namespace {
class RangeBoundariesInferencePass
    : public mlir::bmodelica::impl::RangeBoundariesInferencePassBase<
          RangeBoundariesInferencePass> {
public:
  using RangeBoundariesInferencePassBase::RangeBoundariesInferencePassBase;

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::RewritePatternSet patterns(&getContext());

    patterns.insert<TensorViewOpPattern, TensorInsertSliceOpPattern>(
        &getContext());

    mlir::GreedyRewriteConfig config;
    config.fold = true;

    if (mlir::failed(mlir::applyPatternsGreedily(moduleOp, std::move(patterns),
                                                 config))) {
      moduleOp.emitOpError() << "Can't infer range boundaries";
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createRangeBoundariesInferencePass() {
  return std::make_unique<RangeBoundariesInferencePass>();
}
} // namespace mlir::bmodelica
