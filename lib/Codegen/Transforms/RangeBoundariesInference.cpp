#include "marco/Codegen/Transforms/RangeBoundariesInference.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_RANGEBOUNDARIESINFERENCEPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

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
        mlir::Location loc = op.getLoc();
        llvm::SmallVector<mlir::Value> newIndices;

        for (size_t i = 0, e = op.getIndices().size(); i < e; ++i) {
          mlir::Value index = op.getIndices()[i];

          if (auto unboundedRangeOp =
                  index.getDefiningOp<UnboundedRangeOp>()) {
            mlir::Value lowerBound =
                rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));

            mlir::Value dimensionIndex = rewriter.create<ConstantOp>(
                loc, rewriter.getIndexAttr(static_cast<int64_t>(i)));

            mlir::Value dimensionSize =
                rewriter.create<DimOp>(loc, op.getSource(), dimensionIndex);

            mlir::Value offset =
                rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(-1));

            mlir::Value upperBound = rewriter.create<AddOp>(
                loc, rewriter.getIndexType(), dimensionSize, offset);

            mlir::Value step =
                rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));

            mlir::Value range = rewriter.create<RangeOp>(
                loc,
                RangeType::get(getContext(), rewriter.getIndexType()),
                lowerBound, upperBound, step);

            newIndices.push_back(range);
            continue;
          }

          newIndices.push_back(index);
        }

        rewriter.replaceOpWithNewOp<SubscriptionOp>(
            op, op.getSource(), newIndices);

        return mlir::success();
      }
  };
}

namespace
{
  class RangeBoundariesInferencePass
      : public mlir::modelica::impl::RangeBoundariesInferencePassBase<
            RangeBoundariesInferencePass>
  {
    public:
      using RangeBoundariesInferencePassBase::RangeBoundariesInferencePassBase;

      void runOnOperation() override
      {
        auto moduleOp = getOperation();

        mlir::ConversionTarget target(getContext());
        target.addLegalDialect<ModelicaDialect>();

        target.addDynamicallyLegalOp<SubscriptionOp>([](SubscriptionOp op) {
          return llvm::all_of(op.getIndices(), [](mlir::Value index) {
            mlir::Operation* definingOp = index.getDefiningOp();

            if (!definingOp) {
              return true;
            }

            return !mlir::isa<UnboundedRangeOp>(definingOp);
          });
        });

        mlir::RewritePatternSet patterns(&getContext());

        patterns.insert<SubscriptionOpPattern>(&getContext());

        if (mlir::failed(applyPartialConversion(
                moduleOp, target, std::move(patterns)))) {
          moduleOp.emitOpError() << "Can't infer range boundaries";
          return signalPassFailure();
        }
      }
  };
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createRangeBoundariesInferencePass()
  {
    return std::make_unique<RangeBoundariesInferencePass>();
  }
}
