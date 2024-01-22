#include "marco/Codegen/Transforms/ViewAccessFolding.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_VIEWACCESSFOLDINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class ViewAccessFoldingPass
      : public mlir::modelica::impl::ViewAccessFoldingPassBase<
            ViewAccessFoldingPass>
  {
    public:
      using ViewAccessFoldingPassBase<ViewAccessFoldingPass>
          ::ViewAccessFoldingPassBase;

      void runOnOperation() override;
  };
}

namespace
{
  class RangeViewOnRangeViewPattern
      : public mlir::OpRewritePattern<SubscriptionOp>
  {
    public:
      using mlir::OpRewritePattern<SubscriptionOp>::OpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          SubscriptionOp op, mlir::PatternRewriter& rewriter) const override
      {
        return mlir::failure();
      }
  };
}

void ViewAccessFoldingPass::runOnOperation()
{
  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<RangeViewOnRangeViewPattern>(&getContext());

  mlir::GreedyRewriteConfig config;
  config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;

  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
          getOperation(), std::move(patterns), config))) {
    return signalPassFailure();
  }
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createViewAccessFoldingPass()
  {
    return std::make_unique<ViewAccessFoldingPass>();
  }
}
