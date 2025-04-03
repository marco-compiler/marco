#include <algorithm>
#define DEBUG_TYPE "index-reduction"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/IR/DerivativesMap.h"
#include "marco/Dialect/BaseModelica/IR/Ops.h"
#include "marco/Dialect/BaseModelica/Transforms/IndexReduction.h"
#include "marco/Modeling/IndexReduction.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_INDEXREDUCTIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class IndexReductionPass final
    : public impl::IndexReductionPassBase<IndexReductionPass> {
public:
  using IndexReductionPassBase::IndexReductionPassBase;

  void runOnOperation() override;

private:
  using IndexReductionGraph = marco::modeling::IndexReductionGraph;
  using VariableBridge = mlir::bmodelica::bridge::VariableBridge;
  using EquationBridge = mlir::bmodelica::bridge::EquationBridge;

  mlir::LogicalResult processModelOp(ModelOp modelOp);

  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(EquationTemplateOp equationTemplate,
                            mlir::SymbolTableCollection &symbolTableCollection);

  mlir::LogicalResult initializeGraph(
      IndexReductionGraph &graph,
      llvm::SmallVectorImpl<std::unique_ptr<VariableBridge>> &variableBridges,
      llvm::SmallVectorImpl<std::unique_ptr<EquationBridge>> &equationBridges,
      mlir::SymbolTableCollection &symbolTableCollection,
      llvm::ArrayRef<EquationInstanceOp> mainEquations,
      llvm::ArrayRef<VariableOp> variables,
      const DerivativesMap &derivativesMap);
};
} // namespace

mlir::LogicalResult IndexReductionPass::initializeGraph(
    IndexReductionGraph &graph,
    llvm::SmallVectorImpl<std::unique_ptr<VariableBridge>> &variableBridges,
    llvm::SmallVectorImpl<std::unique_ptr<EquationBridge>> &equationBridges,
    mlir::SymbolTableCollection &symbolTableCollection,
    const llvm::ArrayRef<EquationInstanceOp> mainEquations,
    const llvm::ArrayRef<VariableOp> variables,
    const DerivativesMap &derivativesMap) {
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> variablesMap;

  for (VariableOp variableOp : variables) {
    auto &bridge =
        variableBridges.emplace_back(VariableBridge::build(variableOp));
    graph.addVariable(bridge.get());
    variablesMap[bridge->name] = bridge.get();
  }

  for (EquationInstanceOp equationInstanceOp : mainEquations) {
    auto accessAnalysis = getVariableAccessAnalysis(
        equationInstanceOp.getTemplate(), symbolTableCollection);

    if (!accessAnalysis) {
      equationInstanceOp.emitOpError() << "Can't obtain access analysis";
      return mlir::failure();
    }

    auto &bridge = equationBridges.emplace_back(EquationBridge::build(
        static_cast<int64_t>(equationBridges.size()), equationInstanceOp,
        symbolTableCollection, *accessAnalysis, variablesMap));

    graph.addEquation(bridge.get());
  }

  graph.setDerivatives(derivativesMap);
  return mlir::success();
}

const bool skipArrays = false;

mlir::LogicalResult IndexReductionPass::processModelOp(ModelOp modelOp) {
  mlir::SymbolTableCollection symbolTableCollection;
  mlir::OpBuilder builder(modelOp.getContext());
  builder.setInsertionPointToEnd(modelOp.getBody());

  // Collect equations.
  llvm::SmallVector<EquationInstanceOp> mainEquations;
  modelOp.collectMainEquations(mainEquations);
  // llvm::SmallVector<EquationInstanceOp> initialEquations;
  // modelOp.collectInitialEquations(initialEquations);

  if (skipArrays &&
      llvm::any_of(mainEquations, [](EquationInstanceOp &instanceOp) {
        return !instanceOp.getIterationSpace().empty();
      })) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping model with non-scalar equations\n");
    return mlir::success();
  }

  // Collect variables
  llvm::SmallVector<VariableOp> variables;
  modelOp.collectVariables(variables);
  auto derivativesMap = modelOp.getProperties().getDerivativesMap();

  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::SmallVector<std::unique_ptr<EquationBridge>> equationBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> variablesMap;

  llvm::SmallVector<VariableOp> derivedVariables;

  std::function differentiateVariable =
      [&](const VariableBridge *bridge,
          const IndexSet &indices) -> VariableBridge * {
    std::string derivativeName = bridge->name.getLeafReference().str() + "_d";
    std::string uniqueName = derivativeName;

    mlir::SymbolTable symbolTable =
        symbolTableCollection.getSymbolTable(modelOp);
    int i = 0;
    while (symbolTable.lookup(uniqueName)) {
      uniqueName = derivativeName + "_" + std::to_string(i++);
    }

    mlir::SymbolRefAttr symbolRef =
        mlir::SymbolRefAttr::get(modelOp->getContext(), uniqueName);

    LLVM_DEBUG(llvm::dbgs()
               << "Creating derivative for variable " << bridge->name
               << " with name " << uniqueName << "\n");

    return variableBridges
        .emplace_back(VariableBridge::build(symbolRef, bridge->indices))
        .get();
  };

  std::function differentiateEquation =
      [&](const EquationBridge *bridge) -> EquationBridge * {
    auto id = static_cast<int64_t>(equationBridges.size());
    LLVM_DEBUG(llvm::dbgs() << "Creating derivative for equation "
                            << bridge->getId() << " with id " << id << "\n");
    return equationBridges
        .emplace_back(EquationBridge::build(
            id, bridge->getOp(), symbolTableCollection, variablesMap))
        .get();
  };

  // Build graph
  IndexReductionGraph graph(differentiateVariable, differentiateEquation);

  if (mlir::failed(initializeGraph(graph, variableBridges, equationBridges,
                                   symbolTableCollection, mainEquations,
                                   variables, derivativesMap))) {
    return mlir::failure();
  }

  LLVM_DEBUG(graph.dump(llvm::dbgs()));

  auto res = graph.pantelides();

  LLVM_DEBUG({
    llvm::dbgs() << "Pantelides result:\n";
    size_t index = 0;
    for (const auto &pair : res) {
      llvm::dbgs() << "Equation id: " << pair.first
                   << ", number of derivations: " << pair.second << "\n";
      index = std::max(index, pair.second);
    }

    if (index > 0) {
      llvm::dbgs() << "DAE index: " << index + 1 << "\n";
    } else {
      bool anyMissingDerivative =
          llvm::any_of(variableBridges, [&](auto &bridge) {
            bool hasDerivative =
                derivativesMap.getDerivative(bridge->id).has_value();
            bool isDerivative =
                derivativesMap.getDerivedVariable(bridge->id).has_value();
            return !(hasDerivative || isDerivative);
          });

      if (anyMissingDerivative) {
        llvm::dbgs() << "DAE index: 1\n";
      } else {
        llvm::dbgs() << "DAE index: 0\n";
      }
    }
  });

  for (VariableOp variable : derivedVariables) {
    variable.erase();
  }

  return mlir::success();
}

void IndexReductionPass::runOnOperation() {
  llvm::SmallVector<ModelOp, 1> modelOps;
  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  auto handleModel = [&](mlir::Operation *op) {
    auto modelOp = mlir::cast<ModelOp>(op);

    if (mlir::failed(processModelOp(modelOp))) {
      return mlir::failure();
    }

    return mlir::success();
  };

  if (mlir::failed(mlir::failableParallelForEach(&getContext(), modelOps,
                                                 handleModel))) {
    return signalPassFailure();
  }
}

// Copied from matching pass
std::optional<std::reference_wrapper<VariableAccessAnalysis>>
IndexReductionPass::getVariableAccessAnalysis(
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

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createIndexReductionPass() {
  return std::make_unique<IndexReductionPass>();
}
} // namespace mlir::bmodelica

#undef DEBUG_TYPE
