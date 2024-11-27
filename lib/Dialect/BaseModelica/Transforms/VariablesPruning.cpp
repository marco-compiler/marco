#include "marco/Dialect/BaseModelica/Transforms/VariablesPruning.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/Bridge.h"
#include "marco/Modeling/ArrayEquationsDependencyGraph.h"
#include "marco/Modeling/SingleEntryDigraph.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_VARIABLESPRUNINGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::bridge;

namespace {
class VariablesPruningPass
    : public mlir::bmodelica::impl::VariablesPruningPassBase<
          VariablesPruningPass> {
public:
  using VariablesPruningPassBase<
      VariablesPruningPass>::VariablesPruningPassBase;

  void runOnOperation() override;

private:
  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(EquationTemplateOp equationTemplate,
                            mlir::SymbolTableCollection &symbolTableCollection);

  mlir::LogicalResult processModelOp(ModelOp modelOp);

  mlir::LogicalResult
  collectUsedVariables(llvm::DenseSet<VariableOp> &usedVariables,
                       mlir::SymbolTableCollection &symbolTableCollection,
                       ModelOp modelOp,
                       const llvm::DenseSet<VariableOp> &outputVariables,
                       llvm::ArrayRef<MatchedEquationInstanceOp> equations);

  mlir::LogicalResult
  removeIfUnused(mlir::RewriterBase &rewriter,
                 mlir::SymbolTableCollection &symbolTableCollection,
                 ModelOp modelOp, MatchedEquationInstanceOp equationOp,
                 const llvm::DenseSet<VariableOp> &usedVariables);

  mlir::LogicalResult cleanModelOp(ModelOp modelOp);
};
} // namespace

void VariablesPruningPass::runOnOperation() {
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  auto runFn = [&](mlir::Operation *op) {
    auto modelOp = mlir::cast<ModelOp>(op);

    if (mlir::failed(processModelOp(modelOp))) {
      return mlir::failure();
    }

    if (mlir::failed(cleanModelOp(modelOp))) {
      return mlir::failure();
    }

    return mlir::success();
  };

  if (mlir::failed(
          mlir::failableParallelForEach(&getContext(), modelOps, runFn))) {
    return signalPassFailure();
  }
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
VariablesPruningPass::getVariableAccessAnalysis(
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

mlir::LogicalResult VariablesPruningPass::processModelOp(ModelOp modelOp) {
  mlir::SymbolTableCollection symbolTableCollection;
  mlir::IRRewriter rewriter(&getContext());

  llvm::SmallVector<VariableOp> variableOps;
  modelOp.collectVariables(variableOps);

  // Get the output variables.
  llvm::DenseSet<VariableOp> outputVariables;

  for (VariableOp variableOp : variableOps) {
    if (variableOp.getVariableType().isOutput()) {
      outputVariables.insert(variableOp);
    }
  }

  // Don't perform any optimization if no output variables have been found.
  // The transformation would otherwise result in the elimination of all
  // variables and equations.

  if (outputVariables.empty()) {
    return mlir::success();
  }

  // Collect the equations.
  llvm::SmallVector<MatchedEquationInstanceOp> initialEquations;
  llvm::SmallVector<MatchedEquationInstanceOp> dynamicEquations;
  llvm::SmallVector<MatchedEquationInstanceOp> allEquations;

  modelOp.collectInitialEquations(initialEquations);
  modelOp.collectMainEquations(dynamicEquations);

  allEquations.append(initialEquations);
  allEquations.append(dynamicEquations);

  // Collect the used variables.
  llvm::DenseSet<VariableOp> usedVariables;

  if (mlir::failed(collectUsedVariables(usedVariables, symbolTableCollection,
                                        modelOp, outputVariables,
                                        allEquations))) {
    return mlir::failure();
  }

  // Remove the unneeded equations.
  for (MatchedEquationInstanceOp equationOp : initialEquations) {
    if (mlir::failed(removeIfUnused(rewriter, symbolTableCollection, modelOp,
                                    equationOp, usedVariables))) {
      return mlir::failure();
    }
  }

  for (MatchedEquationInstanceOp equationOp : dynamicEquations) {
    if (mlir::failed(removeIfUnused(rewriter, symbolTableCollection, modelOp,
                                    equationOp, usedVariables))) {
      return mlir::failure();
    }
  }

  // Remove the start operations.
  for (StartOp startOp :
       llvm::make_early_inc_range(modelOp.getOps<StartOp>())) {
    auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
        modelOp, startOp.getVariable());

    if (!usedVariables.contains(variableOp)) {
      rewriter.eraseOp(startOp);
    }
  }

  // Remove the binding equations.
  for (BindingEquationOp bindingEquationOp :
       llvm::make_early_inc_range(modelOp.getOps<BindingEquationOp>())) {
    auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
        modelOp, bindingEquationOp.getVariableAttr());

    if (!usedVariables.contains(variableOp)) {
      rewriter.eraseOp(bindingEquationOp);
    }
  }

  // Remove the unneeded variables.
  for (VariableOp variableOp :
       llvm::make_early_inc_range(modelOp.getVariables())) {
    if (!usedVariables.contains(variableOp)) {
      rewriter.eraseOp(variableOp);
    }
  }

  return mlir::success();
}

namespace {
void walkDerivedVariables(ModelOp modelOp, mlir::SymbolRefAttr variable,
                          std::function<void(mlir::SymbolRefAttr)> callbackFn) {
  const auto &derivativesMap = modelOp.getProperties().derivativesMap;
  auto derivedVar = derivativesMap.getDerivedVariable(variable);

  while (derivedVar) {
    callbackFn(*derivedVar);
    derivedVar = derivativesMap.getDerivedVariable(*derivedVar);
  }
}

void walkDerivativeVariables(
    ModelOp modelOp, mlir::SymbolRefAttr variable,
    std::function<void(mlir::SymbolRefAttr)> callbackFn) {
  const auto &derivativesMap = modelOp.getProperties().derivativesMap;
  auto derivativeVar = derivativesMap.getDerivative(variable);

  while (derivativeVar) {
    callbackFn(*derivativeVar);
    derivativeVar = derivativesMap.getDerivative(*derivativeVar);
  }
}
} // namespace

mlir::LogicalResult VariablesPruningPass::collectUsedVariables(
    llvm::DenseSet<VariableOp> &usedVariables,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    const llvm::DenseSet<VariableOp> &outputVariables,
    llvm::ArrayRef<MatchedEquationInstanceOp> equations) {
  // Create the dependency graph.
  using DependencyGraph = marco::modeling::ArrayEquationsDependencyGraph<
      VariableBridge *, MatchedEquationBridge *,
      marco::modeling::dependency::SingleEntryDigraph<
          marco::modeling::internal::dependency::ArrayEquation<
              MatchedEquationBridge *>>>;

  using DependencyGraphBase = typename DependencyGraph::Base;

  auto baseGraph = std::make_shared<DependencyGraphBase>();
  DependencyGraph graph(&getContext(), baseGraph);

  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> variablesMap;
  llvm::SmallVector<std::unique_ptr<MatchedEquationBridge>> equationBridges;
  llvm::SmallVector<MatchedEquationBridge *> equationPtrs;

  for (VariableOp variableOp : modelOp.getVariables()) {
    auto &bridge =
        variableBridges.emplace_back(VariableBridge::build(variableOp));

    auto symbolRefAttr = mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());
    variablesMap[symbolRefAttr] = bridge.get();
  }

  for (MatchedEquationInstanceOp equation : equations) {
    auto variableAccessAnalysis = getVariableAccessAnalysis(
        equation.getTemplate(), symbolTableCollection);

    auto &bridge = equationBridges.emplace_back(
        MatchedEquationBridge::build(equation, symbolTableCollection,
                                     *variableAccessAnalysis, variablesMap));

    equationPtrs.push_back(bridge.get());
  }

  graph.addEquations(equationPtrs);

  // Add the implicit relations introduced by derivatives.
  for (VariableOp outputVariable : outputVariables) {
    walkDerivedVariables(
        modelOp, mlir::SymbolRefAttr::get(outputVariable.getSymNameAttr()),
        [&](mlir::SymbolRefAttr derivedVarName) {
          auto writingEquations =
              graph.getWritesMap().equal_range(variablesMap[derivedVarName]);

          for (auto writeInfo : llvm::make_range(writingEquations)) {
            baseGraph->addEdge(baseGraph->getEntryNode(),
                               writeInfo.second.getEquation());
          }
        });

    walkDerivativeVariables(
        modelOp, mlir::SymbolRefAttr::get(outputVariable.getSymNameAttr()),
        [&](mlir::SymbolRefAttr derivativeVarName) {
          auto writingEquations =
              graph.getWritesMap().equal_range(variablesMap[derivativeVarName]);

          for (auto writeInfo : llvm::make_range(writingEquations)) {
            baseGraph->addEdge(baseGraph->getEntryNode(),
                               writeInfo.second.getEquation());
          }
        });
  }

  for (auto equationDescriptor :
       llvm::make_range(baseGraph->verticesBegin(), baseGraph->verticesEnd())) {
    const auto &equationBridge = (*baseGraph)[equationDescriptor];

    for (const auto &readAccess : equationBridge.getReads()) {
      mlir::SymbolRefAttr readVariable = readAccess.getVariable()->name;

      walkDerivedVariables(
          modelOp, readVariable, [&](mlir::SymbolRefAttr derivedVarName) {
            auto writingEquations =
                graph.getWritesMap().equal_range(variablesMap[derivedVarName]);

            for (auto writeInfo : llvm::make_range(writingEquations)) {
              baseGraph->addEdge(equationDescriptor,
                                 writeInfo.second.getEquation());
            }
          });

      walkDerivativeVariables(
          modelOp, readVariable, [&](mlir::SymbolRefAttr derivativeVarName) {
            auto writingEquations = graph.getWritesMap().equal_range(
                variablesMap[derivativeVarName]);

            for (auto writeInfo : llvm::make_range(writingEquations)) {
              baseGraph->addEdge(writeInfo.second.getEquation(),
                                 equationDescriptor);
            }
          });
    }
  }

  // Connect the entry node to the output variable.
  for (VariableOp outputVariable : outputVariables) {
    auto variableName =
        mlir::SymbolRefAttr::get(outputVariable.getSymNameAttr());

    auto writingEquations =
        graph.getWritesMap().equal_range(variablesMap[variableName]);

    for (const auto &writeInfo : llvm::make_range(writingEquations)) {
      baseGraph->addEdge(baseGraph->getEntryNode(),
                         writeInfo.second.getEquation());
    }
  }

  // Collect the variables.
  for (auto scc : graph.getSCCs()) {
    for (auto equationDescriptor : scc) {
      auto variableName =
          scc[equationDescriptor].getWrite().getVariable()->name;

      auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
          modelOp, variableName);

      usedVariables.insert(variableOp);

      walkDerivedVariables(
          modelOp, variableName, [&](mlir::SymbolRefAttr derivedVarName) {
            auto derivedVariableOp =
                symbolTableCollection.lookupSymbolIn<VariableOp>(
                    modelOp, derivedVarName);

            usedVariables.insert(derivedVariableOp);
          });

      walkDerivativeVariables(
          modelOp, variableName, [&](mlir::SymbolRefAttr derivativeVarName) {
            auto derivativeVariableOp =
                symbolTableCollection.lookupSymbolIn<VariableOp>(
                    modelOp, derivativeVarName);

            usedVariables.insert(derivativeVariableOp);
          });
    }
  }

  return mlir::success();
}

mlir::LogicalResult VariablesPruningPass::removeIfUnused(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    MatchedEquationInstanceOp equationOp,
    const llvm::DenseSet<VariableOp> &usedVariables) {
  auto matchedAccess = equationOp.getMatchedAccess(symbolTableCollection);

  if (!matchedAccess) {
    return mlir::failure();
  }

  auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
      modelOp, matchedAccess->getVariable());

  if (!usedVariables.contains(variableOp)) {
    rewriter.eraseOp(equationOp);
  }

  return mlir::success();
}

mlir::LogicalResult VariablesPruningPass::cleanModelOp(ModelOp modelOp) {
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  return mlir::applyPatternsAndFoldGreedily(modelOp, std::move(patterns));
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createVariablesPruningPass() {
  return std::make_unique<VariablesPruningPass>();
}
} // namespace mlir::bmodelica
