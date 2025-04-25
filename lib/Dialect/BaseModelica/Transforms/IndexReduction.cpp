#define DEBUG_TYPE "index-reduction"

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/IR/Common.h"
#include "marco/Dialect/BaseModelica/IR/DerivativesMap.h"
#include "marco/Dialect/BaseModelica/IR/Ops.h"
#include "marco/Dialect/BaseModelica/IR/VariableAccess.h"
#include "marco/Dialect/BaseModelica/Transforms/IndexReduction.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/Bridge.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/EquationBridge.h"
#include "marco/Modeling/IndexReduction.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
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

  mlir::LogicalResult
  initializeGraph(IndexReductionGraph &graph, bridge::Storage &storage,
                  mlir::SymbolTableCollection &symbolTableCollection,
                  llvm::ArrayRef<EquationInstanceOp> mainEquations,
                  llvm::ArrayRef<VariableOp> variables,
                  const DerivativesMap &derivativesMap);
};

std::pair<bridge::VariableBridge::Id, std::unique_ptr<AccessFunction>>
convertAccess(const mlir::bmodelica::VariableAccess &access,
              mlir::MLIRContext *context) {
  std::unique_ptr<AccessFunction> accessFunction;
  if (const AccessFunction &initialFunction = access.getAccessFunction();
      initialFunction.getNumOfResults() == 0) {
    // Access to scalar variable.
    accessFunction = AccessFunction::build(
        mlir::AffineMap::get(initialFunction.getNumOfDims(), 0,
                             mlir::getAffineConstantExpr(0, context)));
  } else {
    accessFunction = initialFunction.clone();
  }

  return {access.getVariable(), std::move(accessFunction)};
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

mlir::LogicalResult IndexReductionPass::initializeGraph(
    IndexReductionGraph &graph, bridge::Storage &storage,
    mlir::SymbolTableCollection &symbolTableCollection,
    const llvm::ArrayRef<EquationInstanceOp> mainEquations,
    const llvm::ArrayRef<VariableOp> variables,
    const DerivativesMap &derivativesMap) {
  for (VariableOp variableOp : variables) {
    graph.addVariable(storage.addVariable(variableOp));
  }

  for (EquationInstanceOp equationInstanceOp : mainEquations) {
    auto accessAnalysis = getVariableAccessAnalysis(
        equationInstanceOp.getTemplate(), symbolTableCollection);
    if (!accessAnalysis) {
      equationInstanceOp.emitOpError() << "Can't obtain access analysis";
      return mlir::failure();
    }

    auto &bridge = storage.addEquation(
        static_cast<int64_t>(storage.equationBridges.size()),
        equationInstanceOp, symbolTableCollection);

    auto accesses = llvm::map_to_vector(
        // This cannot be nullopt, as the access analysis is initialized.
        *accessAnalysis->get().getAccesses(symbolTableCollection),
        [&](const auto &access) {
          return convertAccess(access, equationInstanceOp->getContext());
        });
    graph.addEquation(bridge, accesses);
  }

  for (mlir::SymbolRefAttr derivedName : derivativesMap.getDerivedVariables()) {
    mlir::SymbolRefAttr derivativeName =
        *derivativesMap.getDerivative(derivedName);
    IndexSet derivedIndices = *derivativesMap.getDerivedIndices(derivedName);
    if (derivedIndices.empty()) {
      derivedIndices = IndexSet(Point(0));
    }

    auto *derivedBridge = storage.variablesMap[derivedName];
    assert(derivedBridge && "Variable not found");
    auto *derivativeBridge = storage.variablesMap[derivativeName];
    assert(derivativeBridge && "Derivative not found");

    graph.setVariableDerivative(derivedBridge->getId(),
                                derivativeBridge->getId(), derivedIndices);
  }

  return mlir::success();
}

mlir::LogicalResult IndexReductionPass::processModelOp(ModelOp modelOp) {
  mlir::SymbolTableCollection symbolTableCollection;
  mlir::OpBuilder builder(modelOp.getContext());
  builder.setInsertionPointToEnd(modelOp.getBody());

  // Collect equations.
  llvm::SmallVector<EquationInstanceOp> mainEquations;
  modelOp.collectMainEquations(mainEquations);

  // Collect variables
  llvm::SmallVector<VariableOp> variables;
  modelOp.collectVariables(variables);

  DerivativesMap derivativesMap = modelOp.getProperties().getDerivativesMap();
  std::unique_ptr<bridge::Storage> storage = bridge::Storage::create();

  // Callback to build a new differentiated variable from within the graph.
  std::function differentiateVariable =
      [&](const VariableBridge::Id variableId) -> VariableBridge & {
    auto *bridge = storage->variablesMap[variableId];
    assert(bridge && "Variable not found");
    assert(!derivativesMap.getDerivative(bridge->getName()) &&
           "Variable already has a derivative");

    // Create a derivative variable, named after the original variable.
    // Ensure the name is unique in the local scope.
    std::string derivativeName =
        bridge->getName().getLeafReference().str() + "_d";
    auto symbolTable = symbolTableCollection.getSymbolTable(modelOp);
    while (symbolTable.lookup(derivativeName)) {
      derivativeName = "_" + derivativeName;
    }
    mlir::SymbolRefAttr symbolRef =
        mlir::SymbolRefAttr::get(modelOp->getContext(), derivativeName);

    derivativesMap.setDerivative(bridge->getName(), symbolRef);
    return storage->addVariable(symbolRef, bridge->getIndices());
  };

  // Callback to build a new differentiated equation from within the graph.
  std::function differentiateEquation =
      [&](const EquationBridge::Id id) -> EquationBridge & {
    auto *bridge = storage->equationsMap[id];
    assert(bridge && "Equation not found");
    auto derivativeId = static_cast<int64_t>(storage->equationBridges.size());
    return storage->addEquation(derivativeId, bridge->getOp(),
                                symbolTableCollection);
  };

  // Build graph
  IndexReductionGraph graph(differentiateVariable, differentiateEquation);

  if (mlir::failed(initializeGraph(graph, *storage, symbolTableCollection,
                                   mainEquations, variables, derivativesMap))) {
    return mlir::failure();
  }

  LLVM_DEBUG(graph.dump(llvm::dbgs()));

  llvm::SmallVector<std::pair<EquationBridge::Id, llvm::SmallVector<IndexSet>>>
      res = graph.pantelides();

  LLVM_DEBUG({
    llvm::dbgs() << "Pantelides result:\n";
    size_t index = 0;
    for (const auto &[id, derivatives] : res) {
      llvm::dbgs() << "Equation: " << id
                   << " -> #derivations: " << derivatives.size() << " @ ";
      for (size_t i = 0; i < derivatives.size(); i++) {
        llvm::dbgs() << derivatives[i]
                     << (i + 1 < derivatives.size() ? " -> " : "");
      }
      llvm::dbgs() << "\n";
      index = std::max(index, derivatives.size());
    }

    if (index > 0) {
      llvm::dbgs() << "DAE index: " << index + 1 << "\n";
    } else {
      bool anyMissingDerivative =
          llvm::any_of(storage->variableBridges, [&](auto &bridge) {
            bool hasDerivative =
                derivativesMap.getDerivative(bridge->getName()).has_value();
            bool isDerivative =
                derivativesMap.getDerivedVariable(bridge->getName())
                    .has_value();
            return !(hasDerivative || isDerivative);
          });

      if (anyMissingDerivative) {
        llvm::dbgs() << "DAE index: 1\n";
      } else {
        llvm::dbgs() << "DAE index: 0\n";
      }
    }
  });

  if (res.empty()) {
    // The system has index 0 or 1, so no changes are necessary;
    return mlir::success();
  }

  return mlir::success();
}
} // namespace

void IndexReductionPass::runOnOperation() {
  llvm::SmallVector<ModelOp, 1> modelOps;
  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  auto handleModel = [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op);
        mlir::failed(processModelOp(modelOp))) {
      return mlir::failure();
    }

    return mlir::success();
  };

  if (mlir::failed(mlir::failableParallelForEach(&getContext(), modelOps,
                                                 handleModel))) {
    return signalPassFailure();
  }
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createIndexReductionPass() {
  return std::make_unique<IndexReductionPass>();
}
} // namespace mlir::bmodelica

#undef DEBUG_TYPE
