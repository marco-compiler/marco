#include "marco/Dialect/BaseModelica/Transforms/EquationAccessSplit.h"
#include "marco/Dialect/BaseModelica/Analysis/VariableAccessAnalysis.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_EQUATIONACCESSSPLITPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class EquationAccessSplitPass
      : public impl::EquationAccessSplitPassBase<EquationAccessSplitPass>,
        public VariableAccessAnalysis::AnalysisProvider
  {
    public:
      using EquationAccessSplitPassBase<EquationAccessSplitPass>
          ::EquationAccessSplitPassBase;

      void runOnOperation() override;

      std::optional<std::reference_wrapper<VariableAccessAnalysis>>
      getCachedVariableAccessAnalysis(EquationTemplateOp op) override;

    private:
      mlir::LogicalResult processModelOp(ModelOp modelOp);
  };
}

namespace
{
  class EquationOpPattern
      : public mlir::OpRewritePattern<MatchedEquationInstanceOp>
  {
    public:
      EquationOpPattern(
          mlir::MLIRContext* context,
          mlir::SymbolTableCollection& symbolTableCollection)
          : mlir::OpRewritePattern<MatchedEquationInstanceOp>(context),
            symbolTableCollection(&symbolTableCollection)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          MatchedEquationInstanceOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        IndexSet equationIndices = op.getIterationSpace();

        if (equationIndices.empty()) {
          return mlir::failure();
        }

        auto numOfInductions = static_cast<uint64_t>(
            op.getInductionVariables().size());

        std::optional<VariableAccess> matchedAccess =
            op.getMatchedAccess(*symbolTableCollection);

        if (!matchedAccess) {
          return mlir::failure();
        }

        mlir::SymbolRefAttr writtenVariable = matchedAccess->getVariable();

        const AccessFunction& writeAccessFunction =
            matchedAccess->getAccessFunction();

        IndexSet writtenVariableIndices =
            writeAccessFunction.map(equationIndices);

        llvm::SmallVector<VariableAccess> accesses;

        if (mlir::failed(op.getAccesses(accesses, *symbolTableCollection))) {
          return mlir::failure();
        }

        llvm::SmallVector<IndexSet, 10> partitions;
        partitions.push_back(equationIndices);

        for (const VariableAccess& access : accesses) {
          if (access.getPath() == matchedAccess->getPath()) {
            // Ignore the matched access.
            continue;
          }

          const AccessFunction& accessFunction =
              access.getAccessFunction();

          auto readVariable = access.getVariable();

          if (readVariable != writtenVariable) {
            continue;
          }

          IndexSet accessedVariableIndices = accessFunction.map(equationIndices);

          // Restrict the read indices to the written ones.
          accessedVariableIndices =
              accessedVariableIndices.intersect(writtenVariableIndices);

          if (accessedVariableIndices.empty()) {
            // There is no overlap, so there's no need to split the
            // indices of the equation.
            continue;
          }

          // The indices of the equation that lead to a write to the
          // variables reached by the current access.
          IndexSet writingEquationIndices = writeAccessFunction.inverseMap(
              accessedVariableIndices, equationIndices);

          if (writeAccessFunction.isScalarIndependent(accessFunction, writingEquationIndices)) {
            // The accesses overlap only on the array representation, but not on
            // the actual scalar indices.
            continue;
          }

          // Determine the new partitions.
          llvm::SmallVector<IndexSet> newPartitions;

          for (IndexSet& partition : partitions) {
            IndexSet intersection =
                partition.intersect(writingEquationIndices);

            if (intersection.empty()) {
              newPartitions.push_back(std::move(partition));
            } else {
              IndexSet diff = partition - intersection;

              if (!diff.empty()) {
                newPartitions.push_back(std::move(diff));
              }

              newPartitions.push_back(std::move(intersection));
            }
          }

          partitions = std::move(newPartitions);
        }

        // Check that the split indices do represent exactly the initial ones.
        assert(std::accumulate(
                   partitions.begin(), partitions.end(), IndexSet()) ==
               equationIndices);

        if (partitions.size() == 1) {
          return mlir::failure();
        }

        for (const IndexSet& partition : partitions) {
          for (const MultidimensionalRange& range : llvm::make_range(
                   partition.rangesBegin(), partition.rangesEnd())) {
            auto clonedOp = mlir::cast<MatchedEquationInstanceOp>(
                rewriter.clone(*op.getOperation()));

            if (numOfInductions > 0) {
              clonedOp.setIndicesAttr(MultidimensionalRangeAttr::get(
                  getContext(),
                  range.takeFirstDimensions(numOfInductions)));
            }
          }
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }

    private:
      mlir::SymbolTableCollection* symbolTableCollection;
  };
}

void EquationAccessSplitPass::runOnOperation()
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
            return processModelOp(mlir::cast<ModelOp>(op));
          }))) {
    return signalPassFailure();
  }
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
EquationAccessSplitPass::getCachedVariableAccessAnalysis(EquationTemplateOp op)
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::Operation* parentOp = op->getParentOp();
  llvm::SmallVector<mlir::Operation*> parentOps;

  while (parentOp != moduleOp) {
    parentOps.push_back(parentOp);
    parentOp = parentOp->getParentOp();
  }

  mlir::AnalysisManager analysisManager = getAnalysisManager();

  for (mlir::Operation* currentParentOp : llvm::reverse(parentOps)) {
    analysisManager = analysisManager.nest(currentParentOp);
  }

  return analysisManager.getCachedChildAnalysis<VariableAccessAnalysis>(op);
}

mlir::LogicalResult EquationAccessSplitPass::processModelOp(ModelOp modelOp)
{
  mlir::SymbolTableCollection symbolTableCollection;

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<EquationOpPattern>(&getContext(), symbolTableCollection);

  mlir::GreedyRewriteConfig config;
  config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;

  return applyPatternsAndFoldGreedily(modelOp, std::move(patterns), config);
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createEquationAccessSplitPass()
  {
    return std::make_unique<EquationAccessSplitPass>();
  }
}
