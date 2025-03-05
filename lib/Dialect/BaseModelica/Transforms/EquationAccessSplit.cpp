#include "marco/Dialect/BaseModelica/Transforms/EquationAccessSplit.h"
#include "marco/Dialect/BaseModelica/Analysis/VariableAccessAnalysis.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_EQUATIONACCESSSPLITPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class EquationAccessSplitPass
    : public impl::EquationAccessSplitPassBase<EquationAccessSplitPass>,
      public VariableAccessAnalysis::AnalysisProvider {
public:
  using EquationAccessSplitPassBase<
      EquationAccessSplitPass>::EquationAccessSplitPassBase;

  void runOnOperation() override;

  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getCachedVariableAccessAnalysis(EquationTemplateOp op) override;

private:
  mlir::LogicalResult processModelOp(ModelOp modelOp);
};
} // namespace

namespace {
std::optional<size_t>
getMatchedAccessIndex(llvm::ArrayRef<VariableAccess> accesses,
                      const Variable &matchedVariable) {
  std::optional<size_t> result = std::nullopt;

  for (const auto &access : llvm::enumerate(accesses)) {
    if (access.value().getVariable() != matchedVariable.name) {
      continue;
    }

    result = access.index();

    if (access.value().getAccessFunction().isInvertible()) {
      return result;
    }
  }

  return result;
}
} // namespace

namespace {
class EquationOpPattern
    : public mlir::OpRewritePattern<MatchedEquationInstanceOp> {
public:
  EquationOpPattern(mlir::MLIRContext *context,
                    mlir::SymbolTableCollection &symbolTableCollection)
      : mlir::OpRewritePattern<MatchedEquationInstanceOp>(context),
        symbolTableCollection(&symbolTableCollection) {}

  mlir::LogicalResult
  matchAndRewrite(MatchedEquationInstanceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    IndexSet equationIndices = op.getIterationSpace();

    if (equationIndices.empty()) {
      return mlir::failure();
    }

    auto numOfInductions =
        static_cast<uint64_t>(op.getInductionVariables().size());

    const Variable &matchedVariable = op.getProperties().match;
    llvm::SmallVector<VariableAccess> accesses;

    if (mlir::failed(op.getAccesses(accesses, *symbolTableCollection))) {
      return mlir::failure();
    }

    std::optional<size_t> matchedAccessIndex =
        getMatchedAccessIndex(accesses, matchedVariable);

    if (!matchedAccessIndex) {
      return mlir::failure();
    }

    const auto &matchedAccessFunction =
        accesses[*matchedAccessIndex].getAccessFunction();

    llvm::SmallVector<IndexSet, 10> partitions;
    partitions.push_back(equationIndices);

    for (const auto &access : llvm::enumerate(accesses)) {
      if (access.index() == matchedAccessIndex) {
        // Ignore the matched access.
        continue;
      }

      auto accessedVariable = access.value().getVariable();

      if (accessedVariable != matchedVariable.name) {
        continue;
      }

      const AccessFunction &accessFunction = access.value().getAccessFunction();
      IndexSet accessedVariableIndices = accessFunction.map(equationIndices);

      // Restrict the read indices to the written ones.
      accessedVariableIndices =
          accessedVariableIndices.intersect(matchedVariable.indices);

      if (accessedVariableIndices.empty()) {
        // There is no overlap, so there's no need to split the
        // indices of the equation.
        continue;
      }

      // The indices of the equation that lead to a write to the
      // variables reached by the current access.
      IndexSet writingEquationIndices = matchedAccessFunction.inverseMap(
          accessedVariableIndices, equationIndices);

      if (matchedAccessFunction.isScalarIndependent(accessFunction,
                                                    writingEquationIndices)) {
        // The accesses overlap only on the array representation, but not on
        // the actual scalar indices.
        continue;
      }

      // Determine the new partitions.
      llvm::SmallVector<IndexSet> newPartitions;

      for (IndexSet &partition : partitions) {
        IndexSet intersection = partition.intersect(writingEquationIndices);

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
    assert(std::accumulate(partitions.begin(), partitions.end(), IndexSet()) ==
           equationIndices);

    if (partitions.size() == 1) {
      return mlir::failure();
    }

    for (const IndexSet &partition : partitions) {
      for (const MultidimensionalRange &range :
           llvm::make_range(partition.rangesBegin(), partition.rangesEnd())) {
        auto clonedOp = mlir::cast<MatchedEquationInstanceOp>(
            rewriter.clone(*op.getOperation()));

        clonedOp.getProperties().match.indices =
            matchedAccessFunction.map(IndexSet(range));

        if (numOfInductions > 0) {
          clonedOp.setIndicesAttr(MultidimensionalRangeAttr::get(
              getContext(), range.takeFirstDimensions(numOfInductions)));
        }
      }
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  mlir::SymbolTableCollection *symbolTableCollection;
};
} // namespace

void EquationAccessSplitPass::runOnOperation() {
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

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
EquationAccessSplitPass::getCachedVariableAccessAnalysis(
    EquationTemplateOp op) {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::Operation *parentOp = op->getParentOp();
  llvm::SmallVector<mlir::Operation *> parentOps;

  while (parentOp != moduleOp) {
    parentOps.push_back(parentOp);
    parentOp = parentOp->getParentOp();
  }

  mlir::AnalysisManager analysisManager = getAnalysisManager();

  for (mlir::Operation *currentParentOp : llvm::reverse(parentOps)) {
    analysisManager = analysisManager.nest(currentParentOp);
  }

  return analysisManager.getCachedChildAnalysis<VariableAccessAnalysis>(op);
}

mlir::LogicalResult EquationAccessSplitPass::processModelOp(ModelOp modelOp) {
  mlir::SymbolTableCollection symbolTableCollection;

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<EquationOpPattern>(&getContext(), symbolTableCollection);

  mlir::GreedyRewriteConfig config;
  config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;

  return applyPatternsAndFoldGreedily(modelOp, std::move(patterns), config);
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createEquationAccessSplitPass() {
  return std::make_unique<EquationAccessSplitPass>();
}
} // namespace mlir::bmodelica
