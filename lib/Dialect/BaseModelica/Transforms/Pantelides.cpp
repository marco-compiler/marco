#define DEBUG_TYPE "pantelides"

#include "marco/Dialect/BaseModelica/Transforms/Pantelides.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/Bridge.h"
#include "marco/Modeling/Pantelides.h"
#include "llvm/Support/Debug.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_PANTELIDESPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::bridge;

using PantelidesGraph =
    ::marco::modeling::internal::pantelides::PantelidesGraph<VariableBridge *,
                                                             EquationBridge *>;

namespace {
class State {
  std::unique_ptr<PantelidesGraph> graph;
  std::unique_ptr<Storage> storage;
  std::unique_ptr<ad::forward::State> adState;
  llvm::DenseMap<EquationInstanceOp, EquationBridge::Id> equationIds;
  llvm::DenseMap<EquationInstanceOp, EquationInstanceOp> equationDerivatives;

public:
  PantelidesGraph &getGraph() {
    assert(graph && "Graph not set");
    return *graph;
  }

  const PantelidesGraph &getGraph() const {
    assert(graph && "Graph not set");
    return *graph;
  }

  void setGraph(std::unique_ptr<PantelidesGraph> graph) {
    this->graph = std::move(graph);
  }

  Storage &getStorage() {
    assert(storage && "Storage not set");
    return *storage;
  }

  const Storage &getStorage() const {
    assert(storage && "Storage not set");
    return *storage;
  }

  void setStorage(std::unique_ptr<Storage> storage) {
    this->storage = std::move(storage);
  }

  ad::forward::State &getADState() { return *adState; }

  const ad::forward::State &getADState() const { return *adState; }

  void setADState(std::unique_ptr<ad::forward::State> adState) {
    this->adState = std::move(adState);
  }

  void mapEquationToId(EquationInstanceOp equationOp, EquationBridge::Id id) {
    equationIds[equationOp] = id;
  }

  const EquationBridge::Id &getEquationId(EquationInstanceOp equationOp) const {
    auto it = equationIds.find(equationOp);
    assert(it != equationIds.end() && "Equation not found");
    return it->second;
  }

  std::optional<EquationInstanceOp>
  getEquationDerivative(EquationInstanceOp equationOp) {
    if (auto it = equationDerivatives.find(equationOp);
        it != equationDerivatives.end()) {
      return it->second;
    }

    return std::nullopt;
  }

  void setEquationDerivative(EquationInstanceOp equationOp,
                             EquationInstanceOp derivativeEquationOp) {
    equationDerivatives[equationOp] = derivativeEquationOp;
  }
};
} // namespace

namespace {
class PantelidesPass : public impl::PantelidesPassBase<PantelidesPass> {
public:
  using PantelidesPassBase::PantelidesPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult processModelOp(ModelOp modelOp);

  std::unique_ptr<State>
  buildPantelidesGraph(mlir::RewriterBase &rewriter,
                       mlir::SymbolTableCollection &symbolTables,
                       ModelOp modelOp, llvm::ArrayRef<VariableOp> variableOps,
                       llvm::ArrayRef<EquationInstanceOp> equationOps);
};
} // namespace

void PantelidesPass::runOnOperation() {
  llvm::SmallVector<ModelOp, 1> modelOps;
  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), modelOps,
          [&](ModelOp modelOp) { return processModelOp(modelOp); }))) {
    return signalPassFailure();
  }
}

mlir::LogicalResult PantelidesPass::processModelOp(ModelOp modelOp) {
  mlir::IRRewriter rewriter(&getContext());

  // Collect variables and equations.
  llvm::SmallVector<VariableOp> variableOps;
  llvm::SmallVector<EquationInstanceOp> equationOps;

  modelOp.collectVariables(variableOps);
  modelOp.collectMainEquations(equationOps);

  if (equationOps.empty()) {
    return mlir::success();
  }

  mlir::SymbolTableCollection symbolTables;

  // Run the Pantelides algorithm.
  auto state = buildPantelidesGraph(rewriter, symbolTables, modelOp,
                                    variableOps, equationOps);

  if (!state->getGraph().run()) {
    return mlir::failure();
  }

  // TODO: Apply the dummy derivatives algorithm to obtain a square system.

  return mlir::success();
}

namespace {
std::string getDerivativeName(mlir::SymbolRefAttr variableName) {
  std::string result = "der_" + variableName.getRootReference().str();

  for (mlir::FlatSymbolRefAttr component : variableName.getNestedReferences()) {
    result += "." + component.getValue().str();
  }

  return result;
}

void mapVariableDerivative(ad::forward::State &state, VariableOp variableOp,
                           VariableOp derivativeVariableOp) {
  LLVM_DEBUG(
      llvm::dbgs() << "Mapping variable " << derivativeVariableOp.getSymName()
                   << " as derivative of " << variableOp.getSymName() << "\n");

  state.mapGenericOpDerivative(variableOp, derivativeVariableOp);
}

void mapVariableDerivatives(ModelOp modelOp, State &state,
                            mlir::SymbolTableCollection &symbolTables) {
  const DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;

  for (const auto &variable : state.getStorage().getVariables()) {
    auto variableOp =
        symbolTables.lookupSymbolIn<VariableOp>(modelOp, variable->getName());

    if (auto derivative = derivativesMap.getDerivative(variable->getName())) {
      auto derivativeOp =
          symbolTables.lookupSymbolIn<VariableOp>(modelOp, *derivative);

      if (auto derivedIndices =
              derivativesMap.getDerivedIndices(variable->getName());
          !derivedIndices->get().empty()) {
        state.getGraph().addVariableDerivative(variable->getName(), *derivative,
                                               derivedIndices->get());
      } else {
        state.getGraph().addVariableDerivative(variable->getName(), *derivative,
                                               IndexSet(Point(0)));
      }

      mapVariableDerivative(state.getADState(), variableOp, derivativeOp);
    }
  }
}
} // namespace

namespace {
VariableBridge *differentiateVariable(mlir::RewriterBase &rewriter,
                                      mlir::SymbolTable &symbolTable,
                                      DerivativesMap &derivativesMap,
                                      State &state,
                                      VariableBridge *const &variable,
                                      const IndexSet &indices) {
  LLVM_DEBUG(llvm::dbgs() << "Differentiating variable: " << variable->getName()
                          << "\n");

  if (auto derivative = derivativesMap.getDerivative(variable->getName())) {
    LLVM_DEBUG(llvm::dbgs() << "Derivative variable already exists: "
                            << *derivative << "\n");

    // Update the derived indices in the derivatives map.
    if (variable->getOriginalRank() != 0) {
      derivativesMap.addDerivedIndices(variable->getName(), indices);
    }

    // Return the already existing variable.
    assert(state.getStorage().hasVariable(*derivative));
    return &state.getStorage().getVariable(*derivative);
  }

  // Create a new derivative variable.
  auto variableOp =
      symbolTable.lookup<VariableOp>(variable->getName().getRootReference());

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(variableOp);

  auto derVariableOp = rewriter.create<VariableOp>(
      variableOp.getLoc(), getDerivativeName(variable->getName()),
      VariableType::get(variableOp.getVariableType().getShape(),
                        RealType::get(rewriter.getContext()),
                        VariabilityProperty::none, IOProperty::none));

  LLVM_DEBUG(llvm::dbgs() << "Differentiated variable: "
                          << derVariableOp.getSymNameAttr() << "\n");

  symbolTable.insert(derVariableOp, rewriter.getInsertionPoint());

  // Add the variable to the derivatives map.
  derivativesMap.setDerivative(
      variable->getName(),
      mlir::SymbolRefAttr::get(derVariableOp.getSymNameAttr()));

  if (variable->getOriginalRank() != 0) {
    derivativesMap.setDerivedIndices(variable->getName(), indices);
  } else {
    derivativesMap.setDerivedIndices(variable->getName(), {});
  }

  // Add the variable to the AD state.
  mapVariableDerivative(state.getADState(), variableOp, derVariableOp);

  // Add the variable to the graph storage.
  return &state.getStorage().addVariable(derVariableOp);
}

EquationBridge *differentiateEquation(mlir::RewriterBase &rewriter,
                                      mlir::SymbolTableCollection &symbolTables,
                                      State &state,
                                      EquationBridge *const &equation,
                                      const IndexSet &indices) {
  LLVM_DEBUG({
    llvm::dbgs() << "Differentiating equation: ";
    equation->getOp().printInline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  if (auto derivative = state.getEquationDerivative(equation->getOp())) {
    EquationInstanceOp derivedEquationOp = *derivative;

    LLVM_DEBUG({
      llvm::dbgs() << "Derivative equation already exists: ";
      derivedEquationOp.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    // Add the derived indices.
    assert(equation->getOriginalRank() != 0);
    derivedEquationOp.setIndices(derivedEquationOp.getProperties().indices +
                                 indices);

    // Return the already existing equation.
    assert(
        state.getStorage().hasEquation(state.getEquationId(derivedEquationOp)));

    return &state.getStorage().getEquation(
        state.getEquationId(derivedEquationOp));
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(equation->getOp());

  // Derive the equation body.
  EquationTemplateOp derivedTemplateOp =
      ad::forward::createEquationTimeDerivative(
          rewriter, symbolTables, state.getADState(),
          equation->getOp().getTemplate());

  assert(derivedTemplateOp != nullptr &&
         "Can't create time derivative of equation");

  // Create the equation instance.
  rewriter.setInsertionPointAfter(equation->getOp());

  auto instanceOp = rewriter.create<EquationInstanceOp>(
      equation->getOp().getLoc(), derivedTemplateOp);

  if (equation->getOriginalRank() != 0) {
    instanceOp.setIndices(indices);
  }

  state.setEquationDerivative(equation->getOp(), instanceOp);

  LLVM_DEBUG({
    llvm::dbgs() << "Differentiated equation: ";
    instanceOp.printInline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Add the equation to the graph storage.
  auto result = &state.getStorage().addEquation(
      static_cast<int64_t>(state.getStorage().getEquations().size()),
      instanceOp, symbolTables);

  state.mapEquationToId(instanceOp, result->getId());
  return result;
}
} // namespace

std::unique_ptr<State> PantelidesPass::buildPantelidesGraph(
    mlir::RewriterBase &rewriter, mlir::SymbolTableCollection &symbolTables,
    ModelOp modelOp, llvm::ArrayRef<VariableOp> variableOps,
    llvm::ArrayRef<EquationInstanceOp> equationOps) {
  auto state = std::make_unique<State>();
  State *statePtr = state.get();

  // Create the storage for the graph.
  state->setStorage(Storage::create());

  // Create the AD state.
  ad::forward::Options adOptions;
  adOptions.unknowVariablesAsConstants = true;
  state->setADState(std::make_unique<ad::forward::State>(adOptions));

  mlir::SymbolTable &symbolTable = symbolTables.getSymbolTable(modelOp);
  DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;

  auto variableDifferentiationFn =
      [&rewriter, &symbolTable, &derivativesMap,
       statePtr](VariableBridge *const &variable,
                 const IndexSet &indices) -> VariableBridge * {
    return ::differentiateVariable(rewriter, symbolTable, derivativesMap,
                                   *statePtr, variable, indices);
  };

  auto equationDifferentiationFn =
      [&rewriter, &symbolTables,
       statePtr](EquationBridge *const &equation,
                 const IndexSet &indices) -> EquationBridge * {
    return ::differentiateEquation(rewriter, symbolTables, *statePtr, equation,
                                   indices);
  };

  // Create the graph.
  state->setGraph(std::make_unique<PantelidesGraph>(
      &getContext(), variableDifferentiationFn, equationDifferentiationFn));

  // Add the variables to the graph.
  for (VariableOp variableOp : variableOps) {
    if (!variableOp.isReadOnly()) {
      auto &bridge = state->getStorage().addVariable(variableOp);
      state->getGraph().addVariable(&bridge);
    }
  }

  // Map the derivative variables.
  mapVariableDerivatives(modelOp, *state, symbolTables);

  // Add the equations to the graph.
  for (EquationInstanceOp equationOp : equationOps) {
    auto &bridge = state->getStorage().addEquation(
        static_cast<int64_t>(state->getStorage().getEquations().size()),
        equationOp, symbolTables);

    state->mapEquationToId(equationOp, bridge.getId());
    state->getGraph().addEquation(&bridge);
  }

  return state;
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createPantelidesPass() {
  return std::make_unique<PantelidesPass>();
}
} // namespace mlir::bmodelica
