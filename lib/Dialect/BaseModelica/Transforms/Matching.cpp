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

using MatchingGraph =
    ::marco::modeling::MatchingGraph<VariableBridge *, EquationBridge *>;

namespace {
/// Utility class to pack the matching graph with its underlying storage.
class MatchingGraphWrapper {
  MatchingGraph graph;
  std::shared_ptr<Storage> storage;

public:
  MatchingGraphWrapper(MatchingGraph graph, std::shared_ptr<Storage> storage)
      : graph(std::move(graph)), storage(std::move(storage)) {}

  MatchingGraph &operator*() { return graph; }

  const MatchingGraph &operator*() const { return graph; }

  MatchingGraph *operator->() { return &graph; }

  const MatchingGraph *operator->() const { return &graph; }

  Storage &getStorage() {
    assert(storage && "Storage not set");
    return *storage;
  }

  const Storage &getStorage() const {
    assert(storage && "Storage not set");
    return *storage;
  }
};
} // namespace

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

  std::optional<MatchingGraphWrapper>
  buildMatchingGraph(mlir::SymbolTableCollection &symbolTableCollection,
                     llvm::ArrayRef<VariableOp> variableOps,
                     llvm::ArrayRef<EquationInstanceOp> equationOps,
                     llvm::function_ref<std::optional<IndexSet>(VariableOp)>
                         matchableIndicesFn);
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

namespace {
void printMatchingGraph(const MatchingGraph &graph) {
  for (auto vertexDescriptor :
       llvm::make_range(graph.variablesBegin(), graph.variablesEnd())) {
    auto variable = graph.getVariable(vertexDescriptor);
    llvm::errs() << "Variable " << variable.getProperty()->getId() << "\n";
    llvm::errs() << "  - Indices: " << variable.getIndices() << "\n";
    llvm::errs() << "  - Matched indices: " << variable.getMatched() << "\n";
  }

  for (auto vertexDescriptor :
       llvm::make_range(graph.equationsBegin(), graph.equationsEnd())) {
    const auto &equation = graph.getEquation(vertexDescriptor);
    llvm::errs() << "Equation ";
    equation.getProperty()->getOp().printInline(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "  - Indices: " << equation.getIndices() << "\n";
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
  llvm::SmallVector<VariableOp> variables;
  modelOp.collectVariables(variables);

  auto optionalMatchingGraph = buildMatchingGraph(
      symbolTableCollection, variables, equations, matchableIndicesFn);

  if (!optionalMatchingGraph) {
    return mlir::failure();
  }

  auto &matchingGraph = *optionalMatchingGraph;

  auto numberOfScalarEquations = matchingGraph->getNumberOfScalarEquations();
  auto numberOfScalarVariables = matchingGraph->getNumberOfScalarVariables();

  if (numberOfScalarEquations < numberOfScalarVariables) {
    modelOp.emitError() << "Underdetermined model. Found "
                        << numberOfScalarEquations << " scalar equations and "
                        << numberOfScalarVariables << " scalar variables.";

    return mlir::failure();
  }

  if (numberOfScalarEquations > numberOfScalarVariables) {
    modelOp.emitError() << "Overdetermined model. Found "
                        << numberOfScalarEquations << " scalar equations and "
                        << numberOfScalarVariables << " scalar variables.";

    return mlir::failure();
  }

  // Apply the matching algorithm.
  using MatchingSolution = MatchingGraph::MatchingSolution;
  llvm::SmallVector<MatchingSolution> matchingSolutions;

  if (!matchingGraph->match(matchingSolutions, enableSimplificationAlgorithm,
                            enableScalarization, scalarAccessThreshold)) {
    modelOp.emitError()
        << "Matching algorithm could not solve the matching problem";

    printMatchingGraph(*matchingGraph);
    return mlir::failure();
  }

  // Keep track of the old instances to be erased.
  llvm::DenseSet<EquationInstanceOp> toBeErased;

  for (const MatchingSolution &solution : matchingSolutions) {
    // Scalar equations and scalar variables are masked as array equations and
    // array variables, so there should always be some matched indices.
    assert(!solution.getEquationIndices().empty());
    assert(!solution.getVariableIndices().empty());

    auto equationBridge =
        matchingGraph.getStorage().equationsMap[solution.getEquation()];

    auto variableBridge =
        matchingGraph.getStorage().variablesMap[solution.getVariable()];

    size_t equationRank =
        equationBridge->getOp().getInductionVariables().size();

    size_t variableRank = variableBridge->getOriginalIndices().rank();

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(equationBridge->getOp());

    auto matchedEquationOp = rewriter.create<EquationInstanceOp>(
        equationBridge->getOp().getLoc(),
        equationBridge->getOp().getTemplate());

    matchedEquationOp.getProperties() = equationBridge->getOp().getProperties();

    if (equationRank != 0) {
      if (mlir::failed(matchedEquationOp.setIndices(
              solution.getEquationIndices(), symbolTableCollection))) {
        return mlir::failure();
      }
    }

    IndexSet matchedVariableIndices = solution.getVariableIndices();

    if (variableRank == 0) {
      matchedEquationOp.getProperties().match =
          Variable(variableBridge->getName(), {});
    } else {
      matchedEquationOp.getProperties().match =
          Variable(variableBridge->getName(), solution.getVariableIndices());
    }

    // Mark the old instance as obsolete.
    toBeErased.insert(equationBridge->getOp());
  }

  // Erase the old equation instances.
  for (EquationInstanceOp equation : toBeErased) {
    rewriter.eraseOp(equation);
  }

  return mlir::success();
}

std::optional<MatchingGraphWrapper> MatchingPass::buildMatchingGraph(
    mlir::SymbolTableCollection &symbolTableCollection,
    llvm::ArrayRef<VariableOp> variableOps,
    llvm::ArrayRef<EquationInstanceOp> equationOps,
    llvm::function_ref<std::optional<IndexSet>(VariableOp)>
        matchableIndicesFn) {
  MatchingGraph graph(&getContext());
  auto storage = Storage::create();

  for (VariableOp variableOp : variableOps) {
    if (auto indices = matchableIndicesFn(variableOp)) {
      auto symbolRefAttr =
          mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());

      auto &bridge = storage->addVariable(symbolRefAttr, std::move(*indices));
      graph.addVariable(&bridge);
    }
  }

  for (EquationInstanceOp equationOp : equationOps) {
    auto &bridge = storage->addEquation(
        static_cast<int64_t>(storage->equationBridges.size()), equationOp,
        symbolTableCollection);

    if (auto accessAnalysis = getVariableAccessAnalysis(
            equationOp.getTemplate(), symbolTableCollection)) {
      bridge.setAccessAnalysis(*accessAnalysis);
    }

    graph.addEquation(&bridge);
  }

  return MatchingGraphWrapper(std::move(graph), std::move(storage));
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
