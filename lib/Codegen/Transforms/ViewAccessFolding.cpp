#include "marco/Codegen/Transforms/ViewAccessFolding.h"
#include "marco/Dialect/BaseModelica/ModelicaDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_VIEWACCESSFOLDINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class ViewAccessFoldingPass
      : public mlir::bmodelica::impl::ViewAccessFoldingPassBase<
            ViewAccessFoldingPass>
  {
    public:
      using ViewAccessFoldingPassBase<ViewAccessFoldingPass>
          ::ViewAccessFoldingPassBase;

      void runOnOperation() override;
  };
}

static bool isSlicingSubscription(SubscriptionOp op)
{
  int64_t sourceRank = op.getSourceArrayType().getRank();
  auto numOfIndices = static_cast<int64_t>(op.getIndices().size());
  int64_t resultRank = op.getResultArrayType().getRank();
  return sourceRank + numOfIndices != resultRank;
}

static mlir::LogicalResult concatSubscripts(
    llvm::SmallVectorImpl<mlir::Value>& result,
    mlir::OpBuilder& builder,
    mlir::ValueRange firstSubscripts,
    mlir::ValueRange secondSubscripts)
{
  for (mlir::Value subscript : firstSubscripts) {
    if (subscript.getType().isa<IterableTypeInterface>()) {
      if (!subscript.getType().isa<RangeType>()) {
        return mlir::failure();
      }
    }
  }

  size_t pos = 0;

  for (mlir::Value subscript : firstSubscripts) {
    if (subscript.getType().isa<RangeType>()) {
      auto beginOp = builder.create<RangeBeginOp>(
          subscript.getLoc(), subscript);

      auto stepOp = builder.create<RangeStepOp>(
          subscript.getLoc(), subscript);

      mlir::Value secondSubscript = secondSubscripts[pos++];

      auto mulOp = builder.create<MulOp>(
          secondSubscript.getLoc(), builder.getIndexType(),
          secondSubscript, stepOp);

      auto addOp = builder.create<AddOp>(
          secondSubscript.getLoc(), builder.getIndexType(),
          mulOp, beginOp);

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

namespace
{
  class SubscriptionOpPattern
      : public mlir::OpRewritePattern<SubscriptionOp>
  {
    public:
      using mlir::OpRewritePattern<SubscriptionOp>::OpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          SubscriptionOp op, mlir::PatternRewriter& rewriter) const override
      {
        if (isSlicingSubscription(op)) {
          return mlir::failure();
        }

        mlir::Value array = op.getViewSource();
        auto arraySubscriptionOp = array.getDefiningOp<SubscriptionOp>();

        if (!arraySubscriptionOp) {
          return mlir::failure();
        }

        if (!isSlicingSubscription(arraySubscriptionOp)) {
          return mlir::failure();
        }

        llvm::SmallVector<mlir::Value> subscripts;

        if (mlir::failed(concatSubscripts(
                subscripts, rewriter, arraySubscriptionOp.getIndices(),
                op.getIndices()))) {
          return mlir::failure();
        }

        rewriter.replaceOpWithNewOp<SubscriptionOp>(
            op, arraySubscriptionOp.getViewSource(), subscripts);

        return mlir::success();
      }
  };

  class LoadOpPattern : public mlir::OpRewritePattern<LoadOp>
  {
    public:
      using mlir::OpRewritePattern<LoadOp>::OpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          LoadOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Value array = op.getArray();
        auto arraySubscriptionOp = array.getDefiningOp<SubscriptionOp>();

        if (!arraySubscriptionOp) {
          return mlir::failure();
        }

        if (!isSlicingSubscription(arraySubscriptionOp)) {
          return mlir::failure();
        }

        llvm::SmallVector<mlir::Value> subscripts;

        if (mlir::failed(concatSubscripts(
                subscripts, rewriter, arraySubscriptionOp.getIndices(),
                op.getIndices()))) {
          return mlir::failure();
        }

        rewriter.replaceOpWithNewOp<LoadOp>(
            op, arraySubscriptionOp.getViewSource(), subscripts);

        return mlir::success();
      }
  };
}

void ViewAccessFoldingPass::runOnOperation()
{
  mlir::RewritePatternSet patterns(&getContext());

  patterns.insert<
      SubscriptionOpPattern,
      LoadOpPattern>(&getContext());

  RangeBeginOp::getCanonicalizationPatterns(patterns, &getContext());
  RangeStepOp::getCanonicalizationPatterns(patterns, &getContext());

  mlir::GreedyRewriteConfig config;
  config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;

  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
          getOperation(), std::move(patterns), config))) {
    return signalPassFailure();
  }
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createViewAccessFoldingPass()
  {
    return std::make_unique<ViewAccessFoldingPass>();
  }
}
