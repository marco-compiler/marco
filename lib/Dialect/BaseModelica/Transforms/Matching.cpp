#define DEBUG_TYPE "matching"

#include "marco/Dialect/BaseModelica/Transforms/Matching.h"
#include "marco/Dialect/BaseModelica/Analysis/VariableAccessAnalysis.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/Bridge.h"
#include "marco/Modeling/Matching.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_MATCHINGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::bridge;

namespace {
class MatchingPass
    : public mlir::bmodelica::impl::MatchingPassBase<MatchingPass>,
      public VariableAccessAnalysis::AnalysisProvider {
public:
  using MatchingPassBase::MatchingPassBase;

  void runOnOperation() override;

  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getCachedVariableAccessAnalysis(EquationTemplateOp op) override;

private:
  mlir::LogicalResult processModelOp(ModelOp modelOp);

  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(EquationTemplateOp equationTemplate,
                            mlir::SymbolTableCollection &symbolTableCollection);

  mlir::LogicalResult
  match(mlir::IRRewriter &rewriter, ModelOp modelOp,
        llvm::ArrayRef<EquationInstanceOp> equations,
        mlir::SymbolTableCollection &symbolTable,
        llvm::function_ref<std::optional<IndexSet>(VariableOp)>
            matchableIndicesFn);

  mlir::LogicalResult cleanModelOp(ModelOp modelOp);
};
} // namespace

void MatchingPass::runOnOperation() {
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

  // Determine the analyses to be preserved.
  markAnalysesPreserved<DerivativesMap>();
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
MatchingPass::getCachedVariableAccessAnalysis(EquationTemplateOp op) {
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

mlir::LogicalResult MatchingPass::processModelOp(ModelOp modelOp) {
  VariableAccessAnalysis::IRListener variableAccessListener(*this);
  mlir::IRRewriter rewriter(&getContext(), &variableAccessListener);

  // Collect the equations.
  llvm::SmallVector<EquationInstanceOp> initialEquations;
  llvm::SmallVector<EquationInstanceOp> mainEquations;
  modelOp.collectInitialEquations(initialEquations);
  modelOp.collectMainEquations(mainEquations);

  // The symbol table collection to be used for caching.
  mlir::SymbolTableCollection symbolTableCollection;

  // Get the derivatives map.
  const DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;

  // Perform the matching on the 'initial conditions' model.
  if (!initialEquations.empty()) {
    auto matchableIndicesFn =
        [&](VariableOp variable) -> std::optional<IndexSet> {
      return variable.getIndices();
    };

    if (mlir::failed(match(rewriter, modelOp, initialEquations,
                           symbolTableCollection, matchableIndicesFn))) {
      modelOp.emitError()
          << "Matching failed for the 'initial conditions' model";
      return mlir::failure();
    }
  }

  // Perform the matching on the 'main' model.
  if (!mainEquations.empty()) {
    auto matchableIndicesFn =
        [&](VariableOp variable) -> std::optional<IndexSet> {
      if (variable.isReadOnly()) {
        // Read-only variables are handled by initial equations.
        return std::nullopt;
      }

      IndexSet variableIndices = variable.getIndices();

      mlir::SymbolRefAttr variableName =
          mlir::SymbolRefAttr::get(variable.getSymNameAttr());

      if (auto derivedIndices =
              derivativesMap.getDerivedIndices(variableName)) {
        if (variable.getVariableType().isScalar()) {
          return std::nullopt;
        }

        IndexSet result = variableIndices - derivedIndices->get();

        if (result.empty()) {
          return std::nullopt;
        }

        return result;
      }

      return variableIndices;
    };

    if (mlir::failed(match(rewriter, modelOp, mainEquations,
                           symbolTableCollection, matchableIndicesFn))) {
      modelOp.emitError() << "Matching failed for the 'main' model";
      return mlir::failure();
    }
  }

  return mlir::success();
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
MatchingPass::getVariableAccessAnalysis(
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

using MatchingGraph =
    ::marco::modeling::MatchingGraph<VariableBridge *, EquationBridge *>;

namespace {
void printMatchingGraph(const MatchingGraph &graph) {
  for (auto vertexDescriptor :
       llvm::make_range(graph.variablesBegin(), graph.variablesEnd())) {
    const auto &variable = graph.getVariable(vertexDescriptor);
    llvm::errs() << "Variable " << variable.getProperty()->name << "\n";
    llvm::errs() << "  - Indices: " << variable.getIndices() << "\n";
    llvm::errs() << "  - Matched indices: " << variable.getMatched() << "\n";
  }

  for (auto vertexDescriptor :
       llvm::make_range(graph.equationsBegin(), graph.equationsEnd())) {
    const auto &equation = graph.getEquation(vertexDescriptor);
    llvm::errs() << "Equation ";
    equation.getProperty()->op.printInline(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "  - Indices: " << equation.getIterationRanges() << "\n";
    llvm::errs() << "  - Matched indices: " << equation.getMatched() << "\n";
  }
}
} // namespace

mlir::LogicalResult
MatchingPass::match(mlir::IRRewriter &rewriter, ModelOp modelOp,
                    llvm::ArrayRef<EquationInstanceOp> equations,
                    mlir::SymbolTableCollection &symbolTableCollection,
                    llvm::function_ref<std::optional<IndexSet>(VariableOp)>
                        matchableIndicesFn) {
  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> variablesMap;
  llvm::SmallVector<std::unique_ptr<EquationBridge>> equationBridges;

  // Create the matching graph. We use the pointers to the real nodes in order
  // to speed up the copies.
  MatchingGraph matchingGraph(&getContext());

  for (VariableOp variableOp : modelOp.getVariables()) {
    std::optional<IndexSet> indices = matchableIndicesFn(variableOp);

    if (indices) {
      auto symbolRefAttr =
          mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());

      auto &bridge = variableBridges.emplace_back(
          VariableBridge::build(symbolRefAttr, std::move(*indices)));

      variablesMap[symbolRefAttr] = bridge.get();
      matchingGraph.addVariable(bridge.get());
    }
  }

  for (EquationInstanceOp equation : equations) {
    auto accessAnalysis = getVariableAccessAnalysis(equation.getTemplate(),
                                                    symbolTableCollection);

    if (!accessAnalysis) {
      equation.emitOpError() << "Can't obtain access analysis";
      return mlir::failure();
    }

    auto &bridge = equationBridges.emplace_back(EquationBridge::build(
        static_cast<int64_t>(equationBridges.size()), equation,
        symbolTableCollection, *accessAnalysis, variablesMap));

    matchingGraph.addEquation(bridge.get());
  }

  auto numberOfScalarEquations = matchingGraph.getNumberOfScalarEquations();
  auto numberOfScalarVariables = matchingGraph.getNumberOfScalarVariables();

  if (numberOfScalarEquations < numberOfScalarVariables) {
    modelOp.emitError() << "Underdetermined model. Found "
                        << numberOfScalarEquations << " scalar equations and "
                        << numberOfScalarVariables << " scalar variables.";

    return mlir::failure();
  } else if (numberOfScalarEquations > numberOfScalarVariables) {
    modelOp.emitError() << "Overdetermined model. Found "
                        << numberOfScalarEquations << " scalar equations and "
                        << numberOfScalarVariables << " scalar variables.";

    return mlir::failure();
  }

  if (enableSimplificationAlgorithm) {
    // Apply the simplification algorithm to solve the obliged matches.
    if (!matchingGraph.simplify()) {
      modelOp.emitError()
          << "Inconsistency found during the matching simplification process";

      printMatchingGraph(matchingGraph);
      return mlir::failure();
    }
  }

  // Apply the full matching algorithm for the equations and variables that
  // are still unmatched.
  if (!matchingGraph.match()) {
    modelOp.emitError()
        << "Generic matching algorithm could not solve the matching problem";

    printMatchingGraph(matchingGraph);
    return mlir::failure();
  }

  // Keep track of the old instances to be erased.
  llvm::DenseSet<EquationInstanceOp> toBeErased;

  // Get the matching solution.
  using MatchingSolution = MatchingGraph::MatchingSolution;
  llvm::SmallVector<MatchingSolution> matchingSolutions;

  if (!matchingGraph.getMatch(matchingSolutions)) {
    modelOp.emitOpError() << "Not all the equations have been matched";
    return mlir::failure();
  }

  for (const MatchingSolution &solution : matchingSolutions) {
    const IndexSet &matchedEquationIndices = solution.getIndexes();

    // Scalar equations are masked as for-equations, so there should always be
    // some matched index.
    assert(!matchedEquationIndices.empty());

    const EquationPath &matchedPath = solution.getAccess();
    EquationBridge *equation = solution.getEquation();

    std::optional<VariableAccess> matchedAccess =
        equation->op.getAccessAtPath(symbolTableCollection, matchedPath);

    if (!matchedAccess) {
      return mlir::failure();
    }

    size_t numOfInductions = equation->op.getInductionVariables().size();
    bool isScalarEquation = numOfInductions == 0;

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(equation->op);

    auto matchedEquationOp = rewriter.create<MatchedEquationInstanceOp>(
        equation->op.getLoc(), equation->op.getTemplate());

    if (isScalarEquation) {
      matchedEquationOp.getProperties().match.indices =
          matchedAccess->getAccessFunction().map(IndexSet());
    } else {
      IndexSet slicedMatchedIndices =
          matchedEquationIndices.takeFirstDimensions(numOfInductions);

      matchedEquationOp.getProperties().setIndices(slicedMatchedIndices);

      matchedEquationOp.getProperties().match.indices =
          matchedAccess->getAccessFunction().map(slicedMatchedIndices);
    }

    matchedEquationOp.getProperties().match.name = matchedAccess->getVariable();

    // Mark the old instance as obsolete.
    toBeErased.insert(equation->op);
  }

  // Erase the old equation instances.
  for (EquationInstanceOp equation : toBeErased) {
    rewriter.eraseOp(equation);
  }

  return mlir::success();
}

mlir::LogicalResult MatchingPass::cleanModelOp(ModelOp modelOp) {
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  return mlir::applyPatternsAndFoldGreedily(modelOp, std::move(patterns));
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createMatchingPass() {
  return std::make_unique<MatchingPass>();
}

std::unique_ptr<mlir::Pass>
createMatchingPass(const MatchingPassOptions &options) {
  return std::make_unique<MatchingPass>(options);
}
} // namespace mlir::bmodelica
