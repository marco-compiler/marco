#include "marco/Codegen/Transforms/EquationAccessSplit.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_EQUATIONACCESSSPLITPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class EquationAccessSplitPass
      : public impl::EquationAccessSplitPassBase<EquationAccessSplitPass>
  {
    public:
      using EquationAccessSplitPassBase::EquationAccessSplitPassBase;

      void runOnOperation() override;
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

        auto numOfExplicitInductions = static_cast<uint64_t>(
            op.getInductionVariables().size());

        uint64_t numOfImplicitInductions =
            op.getNumOfImplicitInductionVariables();

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
          const AccessFunction& readAccessFunction =
              access.getAccessFunction();

          auto readVariable = access.getVariable();

          if (readVariable != writtenVariable) {
            continue;
          }

          IndexSet accessedVariableIndices =
              readAccessFunction.map(equationIndices);

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

            if (numOfExplicitInductions > 0) {
              clonedOp.setIndicesAttr(MultidimensionalRangeAttr::get(
                  getContext(),
                  range.takeFirstDimensions(numOfExplicitInductions)));
            }

            if (numOfImplicitInductions > 0) {
              clonedOp.setImplicitIndicesAttr(MultidimensionalRangeAttr::get(
                  getContext(),
                  range.takeLastDimensions(numOfImplicitInductions)));
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
  ModelOp modelOp = getOperation();
  mlir::SymbolTableCollection symbolTableCollection;

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<EquationOpPattern>(&getContext(), symbolTableCollection);

  mlir::GreedyRewriteConfig config;
  config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;

  if (mlir::failed(applyPatternsAndFoldGreedily(
          modelOp, std::move(patterns), config))) {
      return signalPassFailure();
  }
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createEquationAccessSplitPass()
  {
    return std::make_unique<EquationAccessSplitPass>();
  }
}
