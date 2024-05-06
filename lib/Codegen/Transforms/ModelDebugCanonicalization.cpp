#include "marco/Codegen/Transforms/ModelDebugCanonicalization.h"
#include "marco/Dialect/BaseModelica/BaseModelicaDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_MODELDEBUGCANONICALIZATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class ModelDebugCanonicalizationPass
      : public mlir::bmodelica::impl::ModelDebugCanonicalizationPassBase<
            ModelDebugCanonicalizationPass>
  {
    public:
      using ModelDebugCanonicalizationPassBase<ModelDebugCanonicalizationPass>
          ::ModelDebugCanonicalizationPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult cleanModel(ModelOp modelOp);

      void sortSCCs(llvm::ArrayRef<SCCOp> SCCs);

      void sortEquations(llvm::ArrayRef<MatchedEquationInstanceOp> equations);
  };
}

void ModelDebugCanonicalizationPass::runOnOperation()
{
  ModelOp modelOp = getOperation();

  // Sort the SCCs and their equations.
  llvm::SmallVector<InitialOp> initialOps;
  llvm::SmallVector<DynamicOp> dynamicOps;

  for (InitialOp initialOp : modelOp.getOps<InitialOp>()) {
    llvm::SmallVector<SCCOp> SCCs;
    initialOp.collectSCCs(SCCs);
    sortSCCs(SCCs);
  }

  for (DynamicOp dynamicOp : modelOp.getOps<DynamicOp>()) {
    llvm::SmallVector<SCCOp> SCCs;
    dynamicOp.collectSCCs(SCCs);
    sortSCCs(SCCs);
  }

  // Clean the IR.
  if (mlir::failed(cleanModel(modelOp))) {
    return signalPassFailure();
  }
}

mlir::LogicalResult ModelDebugCanonicalizationPass::cleanModel(ModelOp modelOp)
{
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  return mlir::applyPatternsAndFoldGreedily(modelOp, std::move(patterns));
}

static MatchedEquationInstanceOp getFirstEquation(SCCOp scc)
{
  if (scc.getBodyRegion().empty()) {
    return nullptr;
  }

  for (auto& op : scc.getBodyRegion().getOps()) {
    if (auto equation = mlir::dyn_cast<MatchedEquationInstanceOp>(op)) {
      return equation;
    }
  }

  return nullptr;
}

void ModelDebugCanonicalizationPass::sortSCCs(llvm::ArrayRef<SCCOp> SCCs)
{
  for (SCCOp scc : SCCs) {
    llvm::SmallVector<MatchedEquationInstanceOp> equations;
    scc.collectEquations(equations);
    sortEquations(equations);
  }

  llvm::SmallVector<SCCOp> sorted(SCCs.begin(), SCCs.end());

  std::sort(
      sorted.begin(), sorted.end(),
      [](SCCOp first, SCCOp second) {
        auto firstEquation = getFirstEquation(first);
        auto secondEquation = getFirstEquation(second);

        if (!secondEquation) {
          return true;
        }

        if (!firstEquation) {
          return false;
        }

        return firstEquation.getTemplate()->isBeforeInBlock(
            secondEquation.getTemplate());
      });

  for (size_t i = 1, e = sorted.size(); i < e; ++i) {
    sorted[i]->moveAfter(sorted[i - 1]);
  }
}

void ModelDebugCanonicalizationPass::sortEquations(
    llvm::ArrayRef<MatchedEquationInstanceOp> equations)
{
  llvm::SmallVector<MatchedEquationInstanceOp> sorted(
      equations.begin(), equations.end());

  std::sort(
      sorted.begin(), sorted.end(),
      [](MatchedEquationInstanceOp first, MatchedEquationInstanceOp second) {
        return first.getTemplate()->isBeforeInBlock(second.getTemplate());
      });

  for (size_t i = 1, e = sorted.size(); i < e; ++i) {
    sorted[i]->moveAfter(sorted[i - 1]);
  }
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createModelDebugCanonicalizationPass()
  {
    return std::make_unique<ModelDebugCanonicalizationPass>();
  }
}
