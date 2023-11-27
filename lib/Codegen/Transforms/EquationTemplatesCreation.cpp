#include "marco/Codegen/Transforms/EquationTemplatesCreation.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <stack>

namespace mlir::modelica
{
#define GEN_PASS_DEF_EQUATIONTEMPLATESCREATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

static mlir::Value collectSubscriptions(
    mlir::Value value,
    llvm::SmallVectorImpl<llvm::SmallVector<mlir::Value, 3>>& subscriptions)
{
  llvm::SmallVector<llvm::SmallVector<mlir::Value, 3>, 6> reverted;
  mlir::Operation* definingOp = value.getDefiningOp();

  auto isValidOp = [](mlir::Operation* op) {
    if (!op) {
      return false;
    }

    return mlir::isa<LoadOp, SubscriptionOp>(op);
  };

  mlir::Value result = value;

  while (isValidOp(definingOp)) {
    if (auto loadOp = mlir::dyn_cast<LoadOp>(definingOp)) {
      for (mlir::Value subscription : loadOp.getIndices()) {
        auto& currentSubscriptions = reverted.emplace_back();
        currentSubscriptions.push_back(subscription);
      }

      result = loadOp.getArray();
      definingOp = result.getDefiningOp();
      continue;
    }

    if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(definingOp)) {
      for (mlir::Value subscription : subscriptionOp.getIndices()) {
        auto& currentSubscriptions = reverted.emplace_back();
        currentSubscriptions.push_back(subscription);
      }

      result = subscriptionOp.getSource();
      definingOp = result.getDefiningOp();
      continue;
    }

    llvm_unreachable("Unexpected operation kind");
  }

  for (auto it = reverted.rbegin(), endIt = reverted.rend();
       it != endIt; ++it) {
    subscriptions.push_back(*it);
  }

  return result;
}

static void reduceSlicedAndFixedSubscriptions(
    llvm::ArrayRef<mlir::Value> subscriptions,
    llvm::SmallVectorImpl<mlir::Value>& reducedSubscriptions)
{
  if (subscriptions.empty()) {
    return;
  }

  reducedSubscriptions.push_back(subscriptions[0]);

  for (size_t i = 1, e = subscriptions.size(); i < e; ++i) {
    if (subscriptions[i].getType().isa<IterableTypeInterface>()) {
      reducedSubscriptions.push_back(subscriptions[i]);
    } else {
      if (!reducedSubscriptions.empty()) {
        reducedSubscriptions.pop_back();
      }
    }
  }
}

static std::vector<MultidimensionalRange> getRangesCombinations(
    size_t rank,
    llvm::function_ref<llvm::ArrayRef<Range>(size_t)> rangesFn,
    size_t startingDimension)
{
  assert(startingDimension < rank);
  std::vector<MultidimensionalRange> result;

  if (startingDimension == rank - 1) {
    for (const Range& range : rangesFn(startingDimension)) {
      result.emplace_back(range);
    }

    return result;
  }

  auto subCombinations = getRangesCombinations(
      rank, rangesFn, startingDimension + 1);

  assert(!rangesFn(startingDimension).empty());

  for (const Range& range : rangesFn(startingDimension)) {
    for (const auto& subCombination : subCombinations) {
      llvm::SmallVector<Range, 1> current;
      current.push_back(range);

      for (size_t i = 0; i < subCombination.rank(); ++i) {
        current.push_back(subCombination[i]);
      }

      result.emplace_back(current);
    }
  }

  return result;
}

static std::vector<MultidimensionalRange> getRangesCombinations(
    size_t rank,
    llvm::function_ref<llvm::ArrayRef<Range>(size_t)> rangesFn)
{
  return getRangesCombinations(rank, rangesFn, 0);
}

namespace
{
  class EquationOpSlicingPattern : public mlir::OpRewritePattern<EquationOp>
  {
    public:
      using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          EquationOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        EquationSidesOp equationSidesOp =
            mlir::cast<EquationSidesOp>(op.getBody()->getTerminator());

        auto lhsValues = equationSidesOp.getLhsValues();
        auto rhsValues = equationSidesOp.getRhsValues();

        assert(lhsValues.size() == rhsValues.size());
        size_t numOfElements = lhsValues.size();

        for (size_t i = 0; i < numOfElements; ++i) {
          llvm::SmallVector<
              llvm::SmallVector<mlir::Value, 3>, 6> lhsSubscriptions;

          llvm::SmallVector<
              llvm::SmallVector<mlir::Value, 3>, 6> rhsSubscriptions;

          mlir::Value lhsRoot =
              collectSubscriptions(lhsValues[i], lhsSubscriptions);

          mlir::Value rhsRoot =
              collectSubscriptions(rhsValues[i], rhsSubscriptions);

          llvm::SmallVector<mlir::Value, 6> lhsReducedSubscriptions;
          llvm::SmallVector<mlir::Value, 6> rhsReducedSubscriptions;

          /*
          reduceSlicedAndFixedSubscriptions(
              lhsSubscriptions, lhsReducedSubscriptions);

          reduceSlicedAndFixedSubscriptions(
              rhsSubscriptions, rhsReducedSubscriptions);
              */

          // Check that we can handle the slicing subscriptions.
          for (mlir::Value subscription : lhsReducedSubscriptions) {
            if (!subscription.getType().isa<RangeType>()) {
              return mlir::failure();
            }
          }

          for (mlir::Value subscription : rhsReducedSubscriptions) {
            if (!subscription.getType().isa<RangeType>()) {
              return mlir::failure();
            }
          }

          assert(lhsReducedSubscriptions.size() ==
                 rhsReducedSubscriptions.size());

          size_t numOfSlicingSubscriptions = lhsReducedSubscriptions.size();
          llvm::SmallVector<mlir::Value, 6> additionalInductions;

          for (size_t j = 0; j < numOfSlicingSubscriptions; ++j) {
            assert(lhsValues[i].getType().cast<ArrayType>().getShape() ==
                   rhsValues[i].getType().cast<ArrayType>().getShape());

            auto elementShape =
                lhsValues[i].getType().cast<ArrayType>().getShape();

            auto forEquationOp = rewriter.create<ForEquationOp>(
                loc, 0, elementShape[j], 1);

            rewriter.setInsertionPointToStart(forEquationOp.bodyBlock());

            additionalInductions.push_back(
                forEquationOp.getBodyRegion().getArgument(0));
          }

          // Clone the equation into the for loop.
          mlir::IRMapping mapping;

          auto clonedEquationOp = mlir::cast<EquationOp>(
              rewriter.clone(*op.getOperation(), mapping));

          // Restrict the equalities to just one result.
          if (mlir::failed(restrictEqualities(
                  rewriter, clonedEquationOp, lhsRoot, rhsRoot,
                  additionalInductions))) {
            return mlir::failure();
          }
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }

    private:
      mlir::LogicalResult restrictEqualities(
        mlir::PatternRewriter& rewriter,
        EquationOp equationOp,
        mlir::Value lhsRoot,
        mlir::Value rhsRoot,
        llvm::ArrayRef<mlir::Value> additionalInductions) const
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);

        auto equationSidesOp = mlir::cast<EquationSidesOp>(
            equationOp.getBody()->getTerminator());

        auto oldLhs = equationSidesOp.getLhs();
        auto oldRhs = equationSidesOp.getRhs();

        rewriter.setInsertionPoint(equationSidesOp);

        mlir::Value lhs = lhsRoot;
        mlir::Value rhs = rhsRoot;

        auto newLhsOp = rewriter.create<EquationSideOp>(oldLhs.getLoc(), lhs);
        auto newRhsOp = rewriter.create<EquationSideOp>(oldRhs.getLoc(), rhs);

        rewriter.replaceOp(oldLhs.getDefiningOp(), newLhsOp);
        rewriter.replaceOp(oldRhs.getDefiningOp(), newRhsOp);

        return mlir::success();
      }
  };

  class EquationOpPattern : public mlir::OpRewritePattern<EquationOp>
  {
    public:
      using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          EquationOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        auto modelOp = op->getParentOfType<ModelOp>();
        rewriter.setInsertionPointToEnd(modelOp.getBody());

        llvm::SmallVector<ForEquationOp, 3> loops;
        getForEquationOps(op, loops);

        llvm::SmallVector<mlir::Value, 3> oldInductions;

        for (ForEquationOp forEquationOp : loops) {
          oldInductions.push_back(forEquationOp.induction());
        }

        // Create the equation template.
        auto templateOp = rewriter.create<EquationTemplateOp>(loc);
        mlir::Block* templateBody = templateOp.createBody(loops.size());

        mlir::IRMapping mapping;

        for (const auto& [oldInduction, newInduction] :
             llvm::zip(oldInductions, templateOp.getInductionVariables())) {
          mapping.map(oldInduction, newInduction);
        }

        rewriter.setInsertionPointToStart(templateBody);

        for (auto& nestedOp : op.getOps()) {
          rewriter.clone(nestedOp, mapping);
        }

        rewriter.setInsertionPointAfter(templateOp);

        // Get the iteration indices.
        IndexSet explicitIndices = getIndices(loops);

        if (mlir::failed(createEquationInstances(
                rewriter, op, templateOp, explicitIndices))) {
          return mlir::failure();
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }

    private:
      void getForEquationOps(
          EquationOp op, llvm::SmallVectorImpl<ForEquationOp>& result) const
      {
        std::stack<ForEquationOp> loopsStack;
        auto parentLoop = op->template getParentOfType<ForEquationOp>();

        while (parentLoop) {
          loopsStack.push(parentLoop);
          parentLoop = parentLoop->template getParentOfType<ForEquationOp>();
        }

        while (!loopsStack.empty()) {
          result.push_back(loopsStack.top());
          loopsStack.pop();
        }
      }

      IndexSet getIndices(
          llvm::ArrayRef<ForEquationOp> loops) const
      {
        if (loops.empty()) {
          return {};
        }

        llvm::SmallVector<llvm::SmallVector<Range, 1>, 3> dimensionsRanges;
        dimensionsRanges.resize(loops.size());

        size_t forEquationIndex = 0;

        for (ForEquationOp forEquationOp : loops) {
          auto from = forEquationOp.getFrom().getSExtValue();
          auto to = forEquationOp.getTo().getSExtValue();
          auto step = forEquationOp.getStep().getSExtValue();

          if (step == 1) {
            dimensionsRanges[forEquationIndex].emplace_back(from, to + 1);
          } else {
            for (auto index = from; index < to + 1; index += step) {
              dimensionsRanges[forEquationIndex].emplace_back(index, index + 1);
            }
          }

          ++forEquationIndex;
        }

        return IndexSet(getRangesCombinations(
            dimensionsRanges.size(),
            [&](size_t dimension) -> llvm::ArrayRef<Range> {
              return dimensionsRanges[dimension];
            }));
      }

      mlir::LogicalResult createEquationInstances(
          mlir::OpBuilder& builder,
          EquationOp equationOp,
          EquationTemplateOp templateOp,
          const IndexSet& explicitIndices) const
      {
        // Compute how many equalities the equation contains.
        auto equationSidesOp = mlir::cast<EquationSidesOp>(
            equationOp.getBody()->getTerminator());

        size_t cardinality = equationSidesOp.getLhsValues().size();

        // Create an instance for each equality.
        for (size_t i = 0; i < cardinality; ++i) {
          std::optional<IndexSet> implicitIndices =
              templateOp.computeImplicitIterationSpace(i);

          if (!explicitIndices.empty()) {
            if (implicitIndices) {
              if (mlir::failed(createInstanceWithExplicitAndImplicitIndices(
                      builder, templateOp, equationOp.getInitial(), i,
                      explicitIndices, *implicitIndices))) {
                return mlir::failure();
              }
            } else {
              if (mlir::failed(createInstanceWithExplicitIndices(
                      builder, templateOp,
                      equationOp.getInitial(), i, explicitIndices))) {
                return mlir::failure();
              }
            }
          } else if (implicitIndices) {
            if (mlir::failed(createInstanceWithImplicitIndices(
                    builder, templateOp,
                    equationOp.getInitial(), i, *implicitIndices))) {
              return mlir::failure();
            }
          } else {
            if (mlir::failed(createScalarInstance(
                    builder, templateOp, equationOp.getInitial(), i))) {
              return mlir::failure();
            }
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult createScalarInstance(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          bool initial,
          uint64_t viewElementIndex) const
      {
        auto instanceOp = builder.create<EquationInstanceOp>(
            templateOp.getLoc(), templateOp, initial);

        if (!instanceOp) {
          return mlir::failure();
        }

        instanceOp.setViewElementIndex(viewElementIndex);
        return mlir::success();
      }

      mlir::LogicalResult createInstanceWithExplicitIndices(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          bool initial,
          uint64_t viewElementIndex,
          const IndexSet& explicitIndices) const
      {
        for (const MultidimensionalRange& range : llvm::make_range(
                 explicitIndices.rangesBegin(), explicitIndices.rangesEnd())) {
          if (mlir::failed(createInstanceWithExplicitIndices(
                  builder, templateOp, initial, viewElementIndex, range))) {
            return mlir::failure();
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult createInstanceWithExplicitIndices(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          bool initial,
          uint64_t viewElementIndex,
          const MultidimensionalRange& explicitIndices) const
      {
        auto instanceOp = builder.create<EquationInstanceOp>(
            templateOp.getLoc(), templateOp, initial);

        instanceOp.setViewElementIndex(viewElementIndex);

        instanceOp.setIndicesAttr(MultidimensionalRangeAttr::get(
            builder.getContext(), explicitIndices));

        return mlir::success();
      }

      mlir::LogicalResult createInstanceWithImplicitIndices(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          bool initial,
          uint64_t viewElementIndex,
          const IndexSet& implicitIndices) const
      {
        for (const MultidimensionalRange& range : llvm::make_range(
                 implicitIndices.rangesBegin(), implicitIndices.rangesEnd())) {
          if (mlir::failed(createInstanceWithImplicitIndices(
                  builder, templateOp, initial, viewElementIndex, range))) {
            return mlir::failure();
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult createInstanceWithImplicitIndices(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          bool initial,
          uint64_t viewElementIndex,
          const MultidimensionalRange& implicitIndices) const
      {
        auto instanceOp = builder.create<EquationInstanceOp>(
            templateOp.getLoc(), templateOp, initial);

        instanceOp.setViewElementIndex(viewElementIndex);

        instanceOp.setImplicitIndicesAttr(MultidimensionalRangeAttr::get(
            builder.getContext(), implicitIndices));

        return mlir::success();
      }

      mlir::LogicalResult createInstanceWithExplicitAndImplicitIndices(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          bool initial,
          uint64_t viewElementIndex,
          const IndexSet& explicitIndices,
          const IndexSet& implicitIndices) const
      {
        for (const MultidimensionalRange& explicitRange : llvm::make_range(
                 explicitIndices.rangesBegin(),
                 explicitIndices.rangesEnd())) {
          for (const MultidimensionalRange& implicitRange : llvm::make_range(
                   implicitIndices.rangesBegin(),
                   implicitIndices.rangesEnd())) {
            if (mlir::failed(createInstanceWithExplicitAndImplicitIndices(
                    builder, templateOp, initial, viewElementIndex,
                    explicitRange, implicitRange))) {
              return mlir::failure();
            }
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult createInstanceWithExplicitAndImplicitIndices(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          bool initial,
          uint64_t viewElementIndex,
          const MultidimensionalRange& explicitIndices,
          const MultidimensionalRange& implicitIndices) const
      {
        auto instanceOp = builder.create<EquationInstanceOp>(
            templateOp.getLoc(), templateOp, initial);

        instanceOp.setViewElementIndex(viewElementIndex);

        instanceOp.setIndicesAttr(MultidimensionalRangeAttr::get(
            builder.getContext(), explicitIndices));

        instanceOp.setImplicitIndicesAttr(MultidimensionalRangeAttr::get(
            builder.getContext(), implicitIndices));

        return mlir::success();
      }
  };

  class ForEquationOpPattern : public mlir::OpRewritePattern<ForEquationOp>
  {
    public:
      using mlir::OpRewritePattern<ForEquationOp>::OpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          ForEquationOp op, mlir::PatternRewriter& rewriter) const override
      {
        if (op.getOps().empty()) {
          rewriter.eraseOp(op);
          return mlir::success();
        }

        return mlir::failure();
      }
  };
}

namespace
{
  class EquationTemplatesCreationPass
      : public impl::EquationTemplatesCreationPassBase<
            EquationTemplatesCreationPass>
  {
    public:
      using EquationTemplatesCreationPassBase
        ::EquationTemplatesCreationPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult explicitateSliceEqualities(ModelOp modelOp);

      mlir::LogicalResult createTemplates(ModelOp modelOp);
  };
}

void EquationTemplatesCreationPass::runOnOperation()
{
  ModelOp modelOp = getOperation();

  /*
  if (mlir::failed(explicitateSliceEqualities(modelOp))) {
      return signalPassFailure();
  }
  */

  if (mlir::failed(createTemplates(modelOp))) {
      return signalPassFailure();
  }
}

mlir::LogicalResult
EquationTemplatesCreationPass::explicitateSliceEqualities(ModelOp modelOp)
{
  mlir::ConversionTarget target(getContext());

  target.addDynamicallyLegalOp<EquationOp>([](EquationOp op) {
    EquationSidesOp equationSidesOp =
        mlir::cast<EquationSidesOp>(op.getBody()->getTerminator());

    auto lhsValues = equationSidesOp.getLhsValues();
    auto rhsValues = equationSidesOp.getRhsValues();

    assert(lhsValues.size() == rhsValues.size());
    size_t numOfElements = lhsValues.size();

    for (size_t i = 0; i < numOfElements; ++i) {
      llvm::SmallVector<llvm::SmallVector<mlir::Value, 3>, 6> lhsSubscriptions;
      llvm::SmallVector<llvm::SmallVector<mlir::Value, 3>, 6> rhsSubscriptions;

      collectSubscriptions(lhsValues[i], lhsSubscriptions);
      collectSubscriptions(rhsValues[i], rhsSubscriptions);

      llvm::SmallVector<mlir::Value, 6> lhsReducedSubscriptions;
      llvm::SmallVector<mlir::Value, 6> rhsReducedSubscriptions;

      /*
      reduceSlicedAndFixedSubscriptions(
          lhsSubscriptions, lhsReducedSubscriptions);

      reduceSlicedAndFixedSubscriptions(
          rhsSubscriptions, rhsReducedSubscriptions);
          */

      if (!lhsReducedSubscriptions.empty() ||
          !rhsReducedSubscriptions.empty()) {
        return false;
      }
    }

    return true;
  });

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<EquationOpSlicingPattern>(&getContext());

  return applyPartialConversion(modelOp, target, std::move(patterns));
}

mlir::LogicalResult
EquationTemplatesCreationPass::createTemplates(ModelOp modelOp)
{
  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<EquationOpPattern, ForEquationOpPattern>(&getContext());

  mlir::GreedyRewriteConfig config;
  config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;

  return applyPatternsAndFoldGreedily(modelOp, std::move(patterns), config);
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createEquationTemplatesCreationPass()
  {
    return std::make_unique<EquationTemplatesCreationPass>();
  }
}
