#include <functional>
#define DEBUG_TYPE "index-reduction"

#include "marco/Dialect/BaseModelica/IR/Ops.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Support/LogicalResult.h"

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/IndexReduction.h"
#include "marco/Modeling/Dumpable.h"
#include "marco/Modeling/IndexReduction.h"

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
  mlir::LogicalResult processModelOp(ModelOp modelOp);

  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(EquationTemplateOp equationTemplate,
                            mlir::SymbolTableCollection &symbolTableCollection);

  void pantelides();

  bool augmenPath();
};
} // namespace

using IndexReductionGraph = marco::modeling::IndexReductionGraph;
using VariableBridge = mlir::bmodelica::bridge::VariableBridge;
using EquationBridge = mlir::bmodelica::bridge::EquationBridge;

mlir::LogicalResult IndexReductionPass::processModelOp(ModelOp modelOp) {
  // Collect equations.
  llvm::SmallVector<EquationInstanceOp> initialEquations;
  llvm::SmallVector<EquationInstanceOp> mainEquations;
  modelOp.collectInitialEquations(initialEquations);
  modelOp.collectMainEquations(mainEquations);

  IndexReductionGraph graph;

  mlir::SymbolTableCollection symbolTableCollection;
  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> variablesMap;
  llvm::SmallVector<std::unique_ptr<EquationBridge>> equationBridges;

  for (VariableOp variableOp : modelOp.getVariables()) {
    auto &bridge = variableBridges.emplace_back(
        VariableBridge::build(variableOp));
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

  LLVM_DEBUG(graph.dump(llvm::dbgs()));

  // For now, only handle main + scalar equations
  if (llvm::any_of(mainEquations, [](EquationInstanceOp &instanceOp) {
        return !instanceOp.getIterationSpace().empty();
      })) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping model with non-scalar equations\n");
    return mlir::success();
  }

  return mlir::success();
}

void IndexReductionPass::pantelides() {}

void IndexReductionPass::runOnOperation() {
  llvm::SmallVector<ModelOp, 1> modelOps;
  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  auto handleModel = [&](mlir::Operation *op) {
    auto modelOp = mlir::cast<ModelOp>(op);
    //LLVM_DEBUG(llvm::dbgs() << "Input model:\n" << modelOp << "\n");

    if (mlir::failed(processModelOp(modelOp))) {
      return mlir::failure();
    }

    //LLVM_DEBUG(llvm::dbgs() << "Output model:\n" << modelOp << "\n");

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
