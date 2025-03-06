#define DEBUG_TYPE "scc-detection"

#include "marco/Dialect/BaseModelica/Transforms/SCCDetection.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/Bridge.h"
#include "marco/Modeling/DependencyGraph.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_SCCDETECTIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::bridge;

namespace {
class SCCDetectionPass
    : public mlir::bmodelica::impl::SCCDetectionPassBase<SCCDetectionPass> {
public:
  using SCCDetectionPassBase<SCCDetectionPass>::SCCDetectionPassBase;

  void runOnOperation() override;

private:
  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(EquationTemplateOp equationTemplate,
                            mlir::SymbolTableCollection &symbolTableCollection);

  mlir::LogicalResult processModelOp(ModelOp modelOp);

  mlir::LogicalResult
  computeSCCs(mlir::IRRewriter &rewriter,
              mlir::SymbolTableCollection &symbolTableCollection,
              ModelOp modelOp, bool initial,
              llvm::ArrayRef<EquationInstanceOp> equations);

  mlir::LogicalResult cleanModelOp(ModelOp modelOp);
};
} // namespace

void SCCDetectionPass::runOnOperation() {
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  auto runFn = [&](mlir::Operation *op) {
    auto modelOp = mlir::cast<ModelOp>(op);
    LLVM_DEBUG(llvm::dbgs() << "Input model:\n" << modelOp << "\n");

    if (mlir::failed(processModelOp(modelOp))) {
      return mlir::failure();
    }

    if (mlir::failed(cleanModelOp(modelOp))) {
      return mlir::failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Output model:\n" << modelOp << "\n");
    return mlir::success();
  };

  if (mlir::failed(
          mlir::failableParallelForEach(&getContext(), modelOps, runFn))) {
    return signalPassFailure();
  }
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
SCCDetectionPass::getVariableAccessAnalysis(
    EquationTemplateOp equationTemplate,
    mlir::SymbolTableCollection &symbolTableCollection) {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::Operation *parentOp = equationTemplate->getParentOp();
  llvm::SmallVector<mlir::Operation *> parentOps;

  while (parentOp != moduleOp) {
    parentOps.push_back(parentOp);
    parentOp = parentOp->getParentOp();
  }

  mlir::AnalysisManager analysisManager = getAnalysisManager();

  for (mlir::Operation *op : llvm::reverse(parentOps)) {
    analysisManager = analysisManager.nest(op);
  }

  if (auto analysis =
          analysisManager.getCachedChildAnalysis<VariableAccessAnalysis>(
              equationTemplate)) {
    return *analysis;
  }

  auto &analysis = analysisManager.getChildAnalysis<VariableAccessAnalysis>(
      equationTemplate);

  if (mlir::failed(analysis.initialize(symbolTableCollection))) {
    return std::nullopt;
  }

  return std::reference_wrapper(analysis);
}

mlir::LogicalResult SCCDetectionPass::processModelOp(ModelOp modelOp) {
  mlir::IRRewriter rewriter(&getContext());

  // Collect the equations.
  llvm::SmallVector<EquationInstanceOp> initialEquations;
  llvm::SmallVector<EquationInstanceOp> mainEquations;

  modelOp.collectInitialEquations(initialEquations);
  modelOp.collectMainEquations(mainEquations);

  // The symbol table collection to be used for caching.
  mlir::SymbolTableCollection symbolTableCollection;

  // Compute the SCCs of the 'initial conditions' model.
  if (!initialEquations.empty()) {
    if (mlir::failed(computeSCCs(rewriter, symbolTableCollection, modelOp, true,
                                 initialEquations))) {
      return mlir::failure();
    }
  }

  // Compute the SCCs of the 'main' model.
  if (!mainEquations.empty()) {
    if (mlir::failed(computeSCCs(rewriter, symbolTableCollection, modelOp,
                                 false, mainEquations))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult SCCDetectionPass::computeSCCs(
    mlir::IRRewriter &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    bool initial, llvm::ArrayRef<EquationInstanceOp> equations) {
  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> variablesMap;
  llvm::SmallVector<std::unique_ptr<EquationBridge>> equationBridges;
  llvm::SmallVector<EquationBridge *> equationPtrs;

  for (VariableOp variableOp : modelOp.getVariables()) {
    auto &bridge =
        variableBridges.emplace_back(VariableBridge::build(variableOp));

    auto symbolRefAttr = mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());
    variablesMap[symbolRefAttr] = bridge.get();
  }

  for (EquationInstanceOp equation : equations) {
    auto variableAccessAnalysis = getVariableAccessAnalysis(
        equation.getTemplate(), symbolTableCollection);

    auto &bridge = equationBridges.emplace_back(EquationBridge::build(
        static_cast<int64_t>(equationBridges.size()), equation,
        symbolTableCollection, *variableAccessAnalysis, variablesMap));

    equationPtrs.push_back(bridge.get());
  }

  using DependencyGraph =
      marco::modeling::DependencyGraph<VariableBridge *, EquationBridge *>;

  DependencyGraph dependencyGraph(&getContext());
  dependencyGraph.addEquations(equationPtrs);

  llvm::SmallVector<DependencyGraph::SCC> SCCs;
  dependencyGraph.getSCCs(SCCs);

  rewriter.setInsertionPointToEnd(modelOp.getBody());

  if (initial) {
    auto initialOp = rewriter.create<InitialOp>(modelOp.getLoc());
    rewriter.createBlock(&initialOp.getBodyRegion());
    rewriter.setInsertionPointToStart(initialOp.getBody());
  } else {
    auto dynamicOp = rewriter.create<DynamicOp>(modelOp.getLoc());
    rewriter.createBlock(&dynamicOp.getBodyRegion());
    rewriter.setInsertionPointToStart(dynamicOp.getBody());
  }

  for (const DependencyGraph::SCC &scc : SCCs) {
    auto sccOp = rewriter.create<SCCOp>(modelOp.getLoc());
    mlir::OpBuilder::InsertionGuard sccGuard(rewriter);

    rewriter.setInsertionPointToStart(
        rewriter.createBlock(&sccOp.getBodyRegion()));

    for (const auto &sccElement : scc) {
      const auto &equation = dependencyGraph[*sccElement];
      const IndexSet &indices = sccElement.getIndices();

      size_t numOfInductions = equation->getOp().getInductionVariables().size();
      bool isScalarEquation = numOfInductions == 0;

      auto clonedOp = mlir::cast<EquationInstanceOp>(
          rewriter.clone(*equation->getOp().getOperation()));

      if (!isScalarEquation) {
        IndexSet slicedIndices = indices.takeFirstDimensions(numOfInductions);

        if (mlir::failed(
                clonedOp.setIndices(slicedIndices, symbolTableCollection))) {
          return mlir::failure();
        }
      }
    }
  }

  for (EquationInstanceOp equation : equations) {
    rewriter.eraseOp(equation);
  }

  return mlir::success();
}

mlir::LogicalResult SCCDetectionPass::cleanModelOp(ModelOp modelOp) {
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  return mlir::applyPatternsAndFoldGreedily(modelOp, std::move(patterns));
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createSCCDetectionPass() {
  return std::make_unique<SCCDetectionPass>();
}
} // namespace mlir::bmodelica
