#include "marco/Dialect/BaseModelica/Transforms/EquationTemplatesCreation.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_EQUATIONTEMPLATESCREATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

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
  class EquationTemplatesCreationPass
      : public impl::EquationTemplatesCreationPassBase<
            EquationTemplatesCreationPass>
  {
    public:
      using EquationTemplatesCreationPassBase<EquationTemplatesCreationPass>
          ::EquationTemplatesCreationPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult createTemplates(ModelOp modelOp);
  };
}

namespace
{
  class EquationOpPattern : public mlir::OpRewritePattern<EquationOp>
  {
    public:
      using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          EquationOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        // Create the equation template.
        mlir::Operation* templateInsertionPoint = op->getParentOp();

        while (templateInsertionPoint &&
               !mlir::isa<DynamicOp, InitialOp>(
                   templateInsertionPoint)) {
          templateInsertionPoint = templateInsertionPoint->getParentOp();
        }

        rewriter.setInsertionPoint(templateInsertionPoint);

        llvm::SmallVector<ForEquationOp, 3> loops;
        getForEquationOps(op, loops);

        llvm::SmallVector<mlir::Value, 3> oldInductions;

        for (ForEquationOp forEquationOp : loops) {
          oldInductions.push_back(forEquationOp.induction());
        }

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

        // Create the equation instances.
        mlir::Operation* instanceInsertionPoint = op.getOperation();

        while (instanceInsertionPoint &&
               instanceInsertionPoint->getParentOp() &&
               !mlir::isa<DynamicOp, InitialOp>(
                   instanceInsertionPoint->getParentOp())) {
          instanceInsertionPoint = instanceInsertionPoint->getParentOp();
        }

        rewriter.setInsertionPoint(instanceInsertionPoint);
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
        llvm::SmallVector<ForEquationOp> loopsStack;
        auto parentLoop = op->template getParentOfType<ForEquationOp>();

        while (parentLoop) {
          loopsStack.push_back(parentLoop);
          parentLoop = parentLoop->template getParentOfType<ForEquationOp>();
        }

        while (!loopsStack.empty()) {
          result.push_back(loopsStack.pop_back_val());
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
        if (!explicitIndices.empty()) {
          if (mlir::failed(createInstanceWithExplicitIndices(
                  builder, templateOp, explicitIndices))) {
            return mlir::failure();
          }
        } else {
          if (mlir::failed(createScalarInstance(builder, templateOp))) {
            return mlir::failure();
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult createScalarInstance(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp) const
      {
        auto instanceOp = builder.create<EquationInstanceOp>(
            templateOp.getLoc(), templateOp);

        if (!instanceOp) {
          return mlir::failure();
        }

        return mlir::success();
      }

      mlir::LogicalResult createInstanceWithExplicitIndices(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          const IndexSet& explicitIndices) const
      {
        for (const MultidimensionalRange& range : llvm::make_range(
                 explicitIndices.rangesBegin(), explicitIndices.rangesEnd())) {
          if (mlir::failed(createInstanceWithExplicitIndices(
                  builder, templateOp, range))) {
            return mlir::failure();
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult createInstanceWithExplicitIndices(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          const MultidimensionalRange& explicitIndices) const
      {
        auto instanceOp = builder.create<EquationInstanceOp>(
            templateOp.getLoc(), templateOp);

        instanceOp.setIndicesAttr(MultidimensionalRangeAttr::get(
            builder.getContext(), explicitIndices));

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

void EquationTemplatesCreationPass::runOnOperation()
{
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation* op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), modelOps,
          [&](mlir::Operation* op) {
            return createTemplates(mlir::cast<ModelOp>(op));
          }))) {
    return signalPassFailure();
  }
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

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createEquationTemplatesCreationPass()
  {
    return std::make_unique<EquationTemplatesCreationPass>();
  }
}
