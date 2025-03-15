#include "marco/Dialect/BaseModelica/Transforms/ViewAccessFolding.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_VIEWACCESSFOLDINGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class ViewAccessFoldingPass
    : public mlir::bmodelica::impl::ViewAccessFoldingPassBase<
          ViewAccessFoldingPass> {
public:
  using ViewAccessFoldingPassBase<
      ViewAccessFoldingPass>::ViewAccessFoldingPassBase;

  void runOnOperation() override;
};
} // namespace

static bool isSlicingSubscription(TensorViewOp op) {
  int64_t sourceRank = op.getSource().getType().getRank();
  auto numOfIndices = static_cast<int64_t>(op.getSubscriptions().size());
  int64_t resultRank = op.getResult().getType().getRank();
  return sourceRank + numOfIndices != resultRank;
}

static mlir::LogicalResult
concatSubscripts(llvm::SmallVectorImpl<mlir::Value> &result,
                 mlir::OpBuilder &builder, mlir::ValueRange firstSubscripts,
                 mlir::ValueRange secondSubscripts) {
  for (mlir::Value subscript : firstSubscripts) {
    if (mlir::isa<IterableTypeInterface>(subscript.getType())) {
      if (!mlir::isa<RangeType>(subscript.getType())) {
        return mlir::failure();
      }
    }
  }

  size_t pos = 0;

  for (mlir::Value subscript : firstSubscripts) {
    if (mlir::isa<RangeType>(subscript.getType())) {
      auto beginOp =
          builder.create<RangeBeginOp>(subscript.getLoc(), subscript);

      auto stepOp = builder.create<RangeStepOp>(subscript.getLoc(), subscript);

      mlir::Value secondSubscript = secondSubscripts[pos++];

      auto mulOp = builder.create<MulOp>(secondSubscript.getLoc(),
                                         builder.getIndexType(),
                                         secondSubscript, stepOp);

      auto addOp = builder.create<AddOp>(
          secondSubscript.getLoc(), builder.getIndexType(), mulOp, beginOp);

      result.push_back(addOp);
    } else {
      result.push_back(subscript);
    }
  }

  for (size_t i = pos, e = secondSubscripts.size(); i < e; ++i) {
    result.push_back(secondSubscripts[i]);
  }

  return mlir::success();
}

namespace {
class TensorViewOpPattern : public mlir::OpRewritePattern<TensorViewOp> {
public:
  using mlir::OpRewritePattern<TensorViewOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TensorViewOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (isSlicingSubscription(op)) {
      return mlir::failure();
    }

    mlir::Value tensor = op.getViewSource();
    auto tensorViewOp = tensor.getDefiningOp<TensorViewOp>();

    if (!tensorViewOp) {
      return mlir::failure();
    }

    if (!isSlicingSubscription(tensorViewOp)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Value> subscripts;

    if (mlir::failed(concatSubscripts(subscripts, rewriter,
                                      tensorViewOp.getSubscriptions(),
                                      op.getSubscriptions()))) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<TensorViewOp>(op, tensorViewOp.getViewSource(),
                                              subscripts);

    return mlir::success();
  }
};

class TensorExtractOpPattern : public mlir::OpRewritePattern<TensorExtractOp> {
public:
  using mlir::OpRewritePattern<TensorExtractOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TensorExtractOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value tensor = op.getTensor();
    auto tensorViewOp = tensor.getDefiningOp<TensorViewOp>();

    if (!tensorViewOp) {
      return mlir::failure();
    }

    if (!isSlicingSubscription(tensorViewOp)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Value> subscripts;

    if (mlir::failed(concatSubscripts(subscripts, rewriter,
                                      tensorViewOp.getSubscriptions(),
                                      op.getIndices()))) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<TensorExtractOp>(
        op, tensorViewOp.getViewSource(), subscripts);

    return mlir::success();
  }
};
} // namespace

void ViewAccessFoldingPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());

  patterns.insert<TensorViewOpPattern, TensorExtractOpPattern>(&getContext());

  RangeBeginOp::getCanonicalizationPatterns(patterns, &getContext());
  RangeStepOp::getCanonicalizationPatterns(patterns, &getContext());

  mlir::GreedyRewriteConfig config;
  config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;
  config.fold = true;

  if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                               std::move(patterns), config))) {
    return signalPassFailure();
  }
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createViewAccessFoldingPass() {
  return std::make_unique<ViewAccessFoldingPass>();
}
} // namespace mlir::bmodelica
