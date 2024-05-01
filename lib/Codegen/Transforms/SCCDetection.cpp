#include "marco/Codegen/Transforms/Modeling/Bridge.h"
#include "marco/Codegen/Transforms/SCCDetection.h"
#include "marco/Dialect/BaseModelica/ModelicaDialect.h"
#include "marco/Modeling/DependencyGraph.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "scc-detection"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_SCCDETECTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::bridge;

namespace
{
  class SCCDetectionPass
      : public mlir::bmodelica::impl::SCCDetectionPassBase<SCCDetectionPass>
  {
    public:
      using SCCDetectionPassBase<SCCDetectionPass>::SCCDetectionPassBase;

      void runOnOperation() override;

    private:
      std::optional<std::reference_wrapper<VariableAccessAnalysis>>
      getVariableAccessAnalysis(
          MatchedEquationInstanceOp equation,
          mlir::SymbolTableCollection& symbolTableCollection);

      mlir::LogicalResult processModelOp(ModelOp modelOp);

      mlir::LogicalResult computeSCCs(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          bool initial,
          llvm::ArrayRef<MatchedEquationInstanceOp> equations);

      mlir::LogicalResult cleanModelOp(ModelOp modelOp);
  };
}

void SCCDetectionPass::runOnOperation()
{
  ModelOp modelOp = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "Input model:\n" << modelOp << "\n");

  if (mlir::failed(processModelOp(modelOp))) {
    return signalPassFailure();
  }

  if (mlir::failed(cleanModelOp(modelOp))) {
    return signalPassFailure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Output model:\n" << modelOp << "\n");
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
SCCDetectionPass::getVariableAccessAnalysis(
    MatchedEquationInstanceOp equation,
    mlir::SymbolTableCollection& symbolTableCollection)
{
  if (auto analysis = getCachedChildAnalysis<VariableAccessAnalysis>(
          equation.getTemplate())) {
    return *analysis;
  }

  auto& analysis = getChildAnalysis<VariableAccessAnalysis>(
      equation.getTemplate());

  if (mlir::failed(analysis.initialize(symbolTableCollection))) {
    return std::nullopt;
  }

  return std::reference_wrapper(analysis);
}

mlir::LogicalResult SCCDetectionPass::processModelOp(ModelOp modelOp)
{
  mlir::IRRewriter rewriter(&getContext());

  // Collect the equations.
  llvm::SmallVector<MatchedEquationInstanceOp> initialEquations;
  llvm::SmallVector<MatchedEquationInstanceOp> mainEquations;

  modelOp.collectInitialEquations(initialEquations);
  modelOp.collectMainEquations(mainEquations);

  // The symbol table collection to be used for caching.
  mlir::SymbolTableCollection symbolTableCollection;

  // Compute the SCCs of the 'initial conditions' model.
  if (!initialEquations.empty()) {
    if (mlir::failed(computeSCCs(
            rewriter, symbolTableCollection, modelOp, true,
            initialEquations))) {
      return mlir::failure();
    }
  }

  // Compute the SCCs of the 'main' model.
  if (!mainEquations.empty()) {
    if (mlir::failed(computeSCCs(
            rewriter, symbolTableCollection, modelOp, false, mainEquations))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult SCCDetectionPass::computeSCCs(
    mlir::IRRewriter& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    bool initial,
    llvm::ArrayRef<MatchedEquationInstanceOp> equations)
{
  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*> variablesMap;
  llvm::SmallVector<std::unique_ptr<MatchedEquationBridge>> equationBridges;
  llvm::SmallVector<MatchedEquationBridge*> equationPtrs;

  for (VariableOp variableOp : modelOp.getVariables()) {
    auto& bridge = variableBridges.emplace_back(
            VariableBridge::build(variableOp));

    auto symbolRefAttr = mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());
    variablesMap[symbolRefAttr] = bridge.get();
  }

  for (MatchedEquationInstanceOp equation : equations) {
    auto variableAccessAnalysis =
        getVariableAccessAnalysis(equation, symbolTableCollection);

    auto& bridge = equationBridges.emplace_back(
        MatchedEquationBridge::build(
            equation, symbolTableCollection, *variableAccessAnalysis,
            variablesMap));

    equationPtrs.push_back(bridge.get());
  }

  using DependencyGraph = marco::modeling::DependencyGraph<
      VariableBridge*, MatchedEquationBridge*>;

  DependencyGraph dependencyGraph(&getContext());
  dependencyGraph.addEquations(equationPtrs);

  llvm::SmallVector<DependencyGraph::SCC> SCCs;
  dependencyGraph.getSCCs(SCCs);

  rewriter.setInsertionPointToEnd(modelOp.getBody());

  if (initial) {
    auto initialModelOp = rewriter.create<InitialModelOp>(modelOp.getLoc());
    rewriter.createBlock(&initialModelOp.getBodyRegion());
    rewriter.setInsertionPointToStart(initialModelOp.getBody());
  } else {
    auto mainModelOp = rewriter.create<MainModelOp>(modelOp.getLoc());
    rewriter.createBlock(&mainModelOp.getBodyRegion());
    rewriter.setInsertionPointToStart(mainModelOp.getBody());
  }

  for (const DependencyGraph::SCC& scc : SCCs) {
    auto sccOp = rewriter.create<SCCOp>(modelOp.getLoc());
    mlir::OpBuilder::InsertionGuard sccGuard(rewriter);

    rewriter.setInsertionPointToStart(
        rewriter.createBlock(&sccOp.getBodyRegion()));

    for (const auto& sccElement : scc) {
      const auto& equation = dependencyGraph[*sccElement];
      const IndexSet& indices = sccElement.getIndices();

      size_t numOfInductions = equation->op.getInductionVariables().size();
      bool isScalarEquation = numOfInductions == 0;

      for (const MultidimensionalRange& matchedEquationRange :
           llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
        auto clonedOp = mlir::cast<MatchedEquationInstanceOp>(
            rewriter.clone(*equation->op.getOperation()));

        if (!isScalarEquation) {
          MultidimensionalRange explicitRange =
              matchedEquationRange.takeFirstDimensions(numOfInductions);

          clonedOp.setIndicesAttr(
              MultidimensionalRangeAttr::get(&getContext(), explicitRange));
        }
      }
    }
  }

  for (MatchedEquationInstanceOp equation : equations) {
    rewriter.eraseOp(equation);
  }

  return mlir::success();
}

mlir::LogicalResult SCCDetectionPass::cleanModelOp(ModelOp modelOp)
{
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  return mlir::applyPatternsAndFoldGreedily(modelOp, std::move(patterns));
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createSCCDetectionPass()
  {
    return std::make_unique<SCCDetectionPass>();
  }
}
