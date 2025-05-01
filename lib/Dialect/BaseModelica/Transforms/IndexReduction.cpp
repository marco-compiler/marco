#include "marco/Dialect/BaseModelica/IR/DerivativesMap.h"
#include "marco/Dialect/BaseModelica/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include <cstddef>
#include <optional>
#include <string>
#define DEBUG_TYPE "index-reduction"

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/IndexReduction.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/Bridge.h"
#include "marco/Modeling/IndexReduction.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_INDEXREDUCTIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
using IndexReductionGraph = marco::modeling::IndexReductionGraph;
using PantelidesResult = IndexReductionGraph::PantelidesResult;

class IndexReductionPass final
    : public impl::IndexReductionPassBase<IndexReductionPass> {
public:
  using IndexReductionPassBase::IndexReductionPassBase;

  void runOnOperation() override;

private:
  using VariableBridge = mlir::bmodelica::bridge::VariableBridge;
  using EquationBridge = mlir::bmodelica::bridge::EquationBridge;

  /// Perform index reduction on the provided model.
  mlir::LogicalResult processModelOp(ModelOp modelOp);

  /// Get the variable access analysis for the given equation template.
  ///
  /// Copied from matching pass.
  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(EquationTemplateOp equationTemplate,
                            mlir::SymbolTableCollection &symbolTableCollection);

  /// Initialize the index reduction graph with the provided equations,
  /// variables, and variable-derivative relationships.
  mlir::LogicalResult
  initializeGraph(IndexReductionGraph &graph, bridge::Storage &storage,
                  mlir::SymbolTableCollection &symbolTableCollection,
                  const llvm::SmallVector<EquationInstanceOp> &equations,
                  const llvm::SmallVector<VariableOp> &variables,
                  const DerivativesMap &derivativesMap);

  /// Run the pantelides algorithm on the provided model.
  mlir::LogicalResult runPantelides(ModelOp modelOp,
                                    PantelidesResult &pantelidesResult,
                                    bridge::Storage &storage);

  /// Create derivative variables and equations based on the pantelides result.
  mlir::LogicalResult
  createDerivatives(ModelOp modelOp, mlir::RewriterBase &rewriter,
                    const PantelidesResult &pantelidesResult,
                    bridge::Storage &storage);
};

/// Convert access to a scalar variable to a constant 0 affine map.
///
/// Copied from matching pass.
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

/// Recursively walk the def-use chain of `op`, deleting all ops where the
/// results are not used.
void eraseDefChain(mlir::Operation *op, mlir::RewriterBase &rewriter) {
  if (!op) {
    return;
  }
  // Only erase if all results are unused
  if (!llvm::all_of(op->getResults(),
                    [](mlir::Value v) { return v.use_empty(); })) {
    return;
  }
  // Recursively erase all operands
  for (mlir::Value operand : op->getOperands()) {
    eraseDefChain(operand.getDefiningOp(), rewriter);
  }
  rewriter.eraseOp(op);
}

/// Replace the operands of the equation side with their derivatives.
mlir::LogicalResult differentiateEquationSide(EquationSideOp equationSideOp,
                                              mlir::RewriterBase &rewriter,
                                              ad::forward::State &state) {
  mlir::SmallVector<mlir::Value> newOperands;
  mlir::SmallVector<mlir::Value> toRemove;
  // An equation side may be a tuple of operands.
  //
  // For each operand: differentiate its defining operation,
  // replace it with the result of the differentiated operation,
  // and erase the original operation.
  for (mlir::Value operand : equationSideOp.getOperands()) {
    auto *defOp = operand.getDefiningOp();
    // The operand might be an induction variable.
    if (!defOp) {
      continue;
    }
    auto derivableOp = llvm::dyn_cast<DerivableOpInterface>(defOp);
    if (!derivableOp) {
      derivableOp->emitOpError("is not derivable");
      return mlir::failure();
    }
    if (mlir::failed(derivableOp.createTimeDerivative(rewriter, state, true))) {
      return mlir::failure();
    }
    newOperands.push_back(*state.getDerivative(operand));
    toRemove.push_back(operand);
  }
  equationSideOp->setOperands(newOperands);
  // Remove the original operands, along with its def-use chain.
  for (mlir::Value operand : toRemove) {
    eraseDefChain(operand.getDefiningOp(), rewriter);
  }
  return mlir::success();
}

mlir::LogicalResult IndexReductionPass::initializeGraph(
    IndexReductionGraph &graph, bridge::Storage &storage,
    mlir::SymbolTableCollection &symbolTableCollection,
    const llvm::SmallVector<EquationInstanceOp> &equations,
    const llvm::SmallVector<VariableOp> &variables,
    const DerivativesMap &derivativesMap) {
  // Collect variable bridges.
  llvm::SmallVector<VariableBridge *> variableBridges;
  for (const VariableOp &variableOp : variables) {
    variableBridges.push_back(&storage.addVariable(variableOp));
  }

  // Collect variable-derivative relationships.
  llvm::SmallVector<IndexReductionGraph::VariableDerivative>
      variableDerivatives;
  for (const mlir::SymbolRefAttr &derivedName :
       derivativesMap.getDerivedVariables()) {
    mlir::SymbolRefAttr derivativeName =
        *derivativesMap.getDerivative(derivedName);
    IndexSet derivedIndices = *derivativesMap.getDerivedIndices(derivedName);
    // Represent scalar variables as having a single 1-D index of 0.
    if (derivedIndices.empty()) {
      derivedIndices = IndexSet(Point(0));
    }

    auto *derivedBridge = storage.variablesMap[derivedName];
    assert(derivedBridge && "Variable not found");
    auto *derivativeBridge = storage.variablesMap[derivativeName];
    assert(derivativeBridge && "Derivative not found");

    variableDerivatives.emplace_back(derivedBridge->getId(),
                                     derivativeBridge->getId(), derivedIndices);
  }

  // Collect equation bridges and their variable-accesses.
  llvm::SmallVector<IndexReductionGraph::EquationWithAccesses> equationBridges;
  for (EquationInstanceOp equationInstanceOp : equations) {
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
        // This always has a value, as the access analysis is initialized.
        *accessAnalysis->get().getAccesses(symbolTableCollection),
        [&](const auto &access) {
          return convertAccess(access, equationInstanceOp->getContext());
        });
    equationBridges.emplace_back(&bridge, std::move(accesses));
  }

  graph.initialize(variableBridges, variableDerivatives, equationBridges);

  return mlir::success();
}

void printPantelidesResult(
    const IndexReductionGraph::PantelidesResult &pantelidesResult,
    const DerivativesMap &derivativesMap, const bridge::Storage &storage) {
  // graph.dump(llvm::dbgs());
  llvm::dbgs() << "Pantelides result:\n";
  size_t index = 0;
  for (const auto &[id, derivatives] : pantelidesResult.equationDerivatives) {
    llvm::dbgs() << " Equation: " << id
                 << " -> #derivations: " << derivatives.size() << " @ ";
    for (size_t i = 0; i < derivatives.size(); i++) {
      llvm::dbgs() << derivatives[i]
                   << (i + 1 < derivatives.size() ? " -> " : "");
    }
    llvm::dbgs() << "\n";
    index = std::max(index, derivatives.size());
  }

  for (const auto &[id, derivatives] : pantelidesResult.variableDerivatives) {
    llvm::dbgs() << " Variable: " << id
                 << " -> #derivations: " << derivatives.size() << " @ ";
    for (size_t i = 0; i < derivatives.size(); i++) {
      llvm::dbgs() << derivatives[i]
                   << (i + 1 < derivatives.size() ? " -> " : "");
    }
    llvm::dbgs() << "\n";
  }

  if (index > 0) {
    llvm::dbgs() << "DAE index: " << index + 1 << "\n";
  } else {
    bool anyMissingDerivative =
        llvm::any_of(storage.variableBridges, [&](auto &bridge) {
          bool hasDerivative =
              derivativesMap.getDerivative(bridge->getName()).has_value();
          bool isDerivative =
              derivativesMap.getDerivedVariable(bridge->getName()).has_value();
          return !(hasDerivative || isDerivative);
        });

    if (anyMissingDerivative) {
      llvm::dbgs() << "DAE index: 1\n";
    } else {
      llvm::dbgs() << "DAE index: 0\n";
    }
  }
}

mlir::LogicalResult
IndexReductionPass::runPantelides(ModelOp modelOp,
                                  PantelidesResult &pantelidesResult,
                                  bridge::Storage &storage) {
  mlir::SymbolTableCollection symbolTableCollection;
  // Callback to build a new differentiated variable from within the graph.
  std::function differentiateVariable =
      [&](const VariableBridge::Id variableId) -> VariableBridge & {
    auto *bridge = storage.variablesMap[variableId];
    assert(bridge && "Variable not found");

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

    return storage.addVariable(symbolRef, bridge->getIndices());
  };

  // Callback to build a new differentiated equation from within the graph.
  std::function differentiateEquation =
      [&](const EquationBridge::Id id) -> EquationBridge & {
    auto *bridge = storage.equationsMap[id];
    assert(bridge && "Equation not found");
    auto derivativeId = static_cast<int64_t>(storage.equationBridges.size());
    return storage.addEquation(derivativeId, bridge->getOp(),
                               symbolTableCollection);
  };

  llvm::SmallVector<EquationInstanceOp> equations;
  llvm::SmallVector<VariableOp> variables;
  modelOp.collectMainEquations(equations);
  modelOp.collectVariables(variables);
  DerivativesMap derivativesMap = modelOp.getDerivativesMap();

  // Build graph
  IndexReductionGraph graph(differentiateVariable, differentiateEquation);
  if (mlir::failed(initializeGraph(graph, storage, symbolTableCollection,
                                   equations, variables, derivativesMap))) {
    return mlir::failure();
  }

  pantelidesResult = graph.pantelides();
  LLVM_DEBUG(printPantelidesResult(pantelidesResult, derivativesMap, storage));

  return mlir::success();
}

mlir::LogicalResult IndexReductionPass::createDerivatives(
    ModelOp modelOp, mlir::RewriterBase &rewriter,
    const PantelidesResult &pantelidesResult, bridge::Storage &storage) {
  DerivativesMap derivativesMap = modelOp.getDerivativesMap();
  mlir::SymbolTable symbolTable(modelOp);
  ad::forward::State state;

  rewriter.setInsertionPointToEnd(modelOp.getBody());
  for (const auto &[variable, derivatives] :
       pantelidesResult.variableDerivatives) {
    mlir::SymbolRefAttr variableName =
        storage.variablesMap[variable]->getName();
    VariableOp current = mlir::cast<VariableOp>(
        symbolTable.lookupSymbolIn(modelOp, variableName));

    // Handle each prescribed derivation.
    for (const IndexSet &toDerive : derivatives) {
      mlir::Operation *derivative = nullptr;
      auto currentNameAttr = mlir::SymbolRefAttr::get(current.getSymNameAttr());

      if (auto existingDerivative =
              derivativesMap.getDerivative(currentNameAttr)) {
        // If the variable has a derivative use it, possibly extending its
        // indices.
        const IndexSet &alreadyDerivedIndices =
            *derivativesMap.getDerivedIndices(currentNameAttr);
        if (!alreadyDerivedIndices.contains(toDerive)) {
          derivativesMap.addDerivedIndices(currentNameAttr, toDerive);
        }

        derivative = symbolTable.lookupSymbolIn(modelOp, *existingDerivative);
      } else {
        // Create a new derivative variable
        std::string derivativeName = "der_" + current.getName().str();
        VariableOp derivativeOp = rewriter.create<VariableOp>(
            rewriter.getUnknownLoc(), derivativeName,
            current.getVariableType());
        derivativesMap.setDerivative(
            currentNameAttr,
            mlir::SymbolRefAttr::get(derivativeOp.getSymNameAttr()));
        derivativesMap.addDerivedIndices(currentNameAttr, toDerive);
        derivative = derivativeOp;
      }

      state.mapGenericOpDerivative(current, derivative);
      current = mlir::cast<VariableOp>(derivative);
    }
  }

  // Update the model with the newly added derivatives.
  modelOp.setDerivativesMap(derivativesMap);

  llvm::SmallVector<std::pair<EquationTemplateOp, std::optional<IndexSet>>>
      equationDerivatives;
  for (const auto &[equationId, derivatives] :
       pantelidesResult.equationDerivatives) {
    mlir::IRMapping mapping;
    auto current = mlir::cast<EquationTemplateOp>(
        storage.equationsMap[equationId]->getOp().getTemplate());

    for (const IndexSet &toDerive : derivatives) {
      rewriter.setInsertionPointToEnd(modelOp.getBody());

      auto derivativeTemplateOp =
          mlir::cast<EquationTemplateOp>(rewriter.clone(*current, mapping));

      auto equationSidesOp = mlir::cast<EquationSidesOp>(
          derivativeTemplateOp.getBody()->getTerminator());

      rewriter.setInsertionPointToStart(derivativeTemplateOp.getBody());

      if (mlir::failed(differentiateEquationSide(
              mlir::cast<EquationSideOp>(
                  equationSidesOp.getLhs().getDefiningOp<EquationSideOp>()),
              rewriter, state))) {
        return mlir::failure();
      }

      if (mlir::failed(differentiateEquationSide(
              mlir::cast<EquationSideOp>(
                  equationSidesOp.getRhs().getDefiningOp<EquationSideOp>()),
              rewriter, state))) {
        return mlir::failure();
      }

      equationDerivatives.emplace_back(
          derivativeTemplateOp,
          derivativeTemplateOp.getInductionVariables().empty()
              ? std::nullopt
              : std::make_optional(toDerive));

      current = derivativeTemplateOp;
    }
  }

  rewriter.setInsertionPointToEnd(modelOp.getBody());
  auto dynamicOp = rewriter.create<DynamicOp>(modelOp.getLoc());
  rewriter.createBlock(&dynamicOp.getBodyRegion());
  rewriter.setInsertionPointToStart(dynamicOp.getBody());
  for (auto &[derivativeTemplateOp, indices] : equationDerivatives) {
    if (indices) {
      rewriter.create<EquationInstanceOp>(rewriter.getUnknownLoc(),
                                          derivativeTemplateOp, *indices);
    } else {
      rewriter.create<EquationInstanceOp>(rewriter.getUnknownLoc(),
                                          derivativeTemplateOp);
    }
  }

  return mlir::success();
}

mlir::LogicalResult IndexReductionPass::processModelOp(ModelOp modelOp) {
  std::unique_ptr<bridge::Storage> storage = bridge::Storage::create();
  PantelidesResult pantelidesResult;
  if (mlir::failed(runPantelides(modelOp, pantelidesResult, *storage))) {
    return mlir::failure();
  }

  // If no equation derivatives are prescribed, the system has index 0 or 1, so
  // no changes are necessary.
  if (pantelidesResult.equationDerivatives.empty()) {
    return mlir::success();
  }

  // Create derivative equations and variables.
  mlir::IRRewriter rewriter(modelOp);
  if (mlir::failed(
          createDerivatives(modelOp, rewriter, pantelidesResult, *storage))) {
  }

  // At this point the dummy derivatives algorithm should be applied to
  // rebalance the model. Removing one of the newly added derivative variables
  // per new derivative equation.

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
