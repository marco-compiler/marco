#include "marco/Dialect/BaseModelica/Transforms/CallCSE.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

#include <marco/AST/Node/Operation.h>

namespace mlir::bmodelica {
#define GEN_PASS_DEF_CALLCSEPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class CallCSEPass final : public impl::CallCSEPassBase<CallCSEPass> {
public:
  using CallCSEPassBase::CallCSEPassBase;

  void runOnOperation() override;

private:
  static mlir::LogicalResult processModelOp(ModelOp modelOp);

  /// Get all callOps in the model.
  static void collectCallOps(ModelOp modelOp,
                             llvm::SmallVector<CallOp> &callOps);

  /// Partition the list of call operations into groups given by
  /// EquationExpressionOpInterface::isEquivalent
  static void buildCallEquivalenceGroups(
      llvm::SmallVector<CallOp> &callOps,
      mlir::SymbolTableCollection &symbolTableCollection,
      llvm::SmallVector<llvm::SmallVector<CallOp>> &callEquivalenceGroups);

  static mlir::Operation *cloneDefUseChain(mlir::Operation *op,
                                           mlir::IRMapping &mapping,
                                           mlir::RewriterBase &rewriter);

  /// Replace all calls in the equivalence group with gets to a generated
  /// variable. The variable will be driven by an equation derived from the
  /// first call in the group.
  static EquationTemplateOp emitCse(ModelOp modelOp, int emittedCSEs,
                                    llvm::SmallVector<CallOp> &equivalenceGroup,
                                    mlir::RewriterBase &rewriter);
};
} // namespace

void CallCSEPass::runOnOperation() {
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  if (failed(failableParallelForEach(
          &getContext(), modelOps, [&](mlir::Operation *op) {
            return processModelOp(mlir::cast<ModelOp>(op));
          }))) {
    return signalPassFailure();
  }
}

void CallCSEPass::collectCallOps(ModelOp modelOp,
                                 llvm::SmallVector<CallOp> &callOps) {
  llvm::SmallVector<EquationInstanceOp> initialEquationOps;
  llvm::SmallVector<EquationInstanceOp> dynamicEquationOps;

  modelOp.collectInitialEquations(initialEquationOps);
  modelOp.collectMainEquations(dynamicEquationOps);

  llvm::DenseSet<EquationTemplateOp> templateOps;

  // TODO: Figure out if these should be included
  // for (auto equationOp : initialEquationOps) {
  //   templateOps.insert(equationOp.getTemplate());
  // }

  for (auto equationOp : dynamicEquationOps) {
    templateOps.insert(equationOp.getTemplate());
  }

  for (auto templateOp : templateOps) {
    // Skip templates with induction variables
    if (!templateOp.getInductionVariables().empty()) {
      continue;
    }
    templateOp->walk([&](CallOp callOp) { callOps.push_back(callOp); });
  }
}

void CallCSEPass::buildCallEquivalenceGroups(
    llvm::SmallVector<CallOp> &callOps,
    mlir::SymbolTableCollection &symbolTableCollection,
    llvm::SmallVector<llvm::SmallVector<CallOp>> &callEquivalenceGroups) {
  for (auto callOp : callOps) {
    auto callExpression =
        mlir::cast<EquationExpressionOpInterface>(callOp.getOperation());

    auto *equivalenceGroup =
        find_if(callEquivalenceGroups, [&](llvm::SmallVector<CallOp> &group) {
          // front() is safe as there are no empty groups
          auto representative = mlir::cast<EquationExpressionOpInterface>(
              group.front().getOperation());
          return callExpression.isEquivalent(representative,
                                             symbolTableCollection);
        });

    if (equivalenceGroup != callEquivalenceGroups.end()) {
      // Add equivalent call to existing group
      equivalenceGroup->push_back(callOp);
    } else {
      // Create new equivalence group
      callEquivalenceGroups.push_back({callOp});
    }
  }
}

mlir::Operation *CallCSEPass::cloneDefUseChain(mlir::Operation *op,
                                               mlir::IRMapping &mapping,
                                               mlir::RewriterBase &rewriter) {
  std::vector<mlir::Operation *> toClone;
  std::vector worklist({op});

  // DFS through the def-use chain of `op`
  while (!worklist.empty()) {
    auto *current = worklist.back();
    worklist.pop_back();
    toClone.push_back(current);
    for (auto operand : current->getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        worklist.push_back(defOp);
      }
    }
  }

  mlir::Operation *root = nullptr;
  for (auto *opToClone : llvm::reverse(toClone)) {
    root = rewriter.clone(*opToClone, mapping);
  }
  return root;
}

EquationTemplateOp
CallCSEPass::emitCse(ModelOp modelOp, const int emittedCSEs,
                     llvm::SmallVector<CallOp> &equivalenceGroup,
                     mlir::RewriterBase &rewriter) {
  assert(!equivalenceGroup.empty() && "equivalenceGroup cannot be empty");
  auto representative = equivalenceGroup.front();
  const auto loc = representative.getLoc();
  // Emit CSE variable
  rewriter.setInsertionPointToStart(modelOp.getBody());
  auto cseVariable = rewriter.create<VariableOp>(
      loc, "_cse" + std::to_string(emittedCSEs),
      VariableType::wrap(representative.getResult(0).getType()));

  // Create CSE variable driver equation
  rewriter.setInsertionPointToEnd(modelOp.getBody());
  auto equationTemplateOp = rewriter.create<EquationTemplateOp>(loc);
  rewriter.setInsertionPointToStart(equationTemplateOp.createBody(0));

  mlir::IRMapping mapping;
  auto *clonedRepresentative =
      cloneDefUseChain(representative, mapping, rewriter);

  auto cseGetOp = rewriter.create<VariableGetOp>(loc, cseVariable);
  auto lhsOp = rewriter.create<EquationSideOp>(loc, cseGetOp->getResults());
  auto rhsOp =
      rewriter.create<EquationSideOp>(loc, clonedRepresentative->getResults());
  rewriter.create<EquationSidesOp>(loc, lhsOp, rhsOp);

  // Replace calls with get to CSE variable
  for (auto &callOp : equivalenceGroup) {
    rewriter.setInsertionPoint(callOp);
    rewriter.replaceOpWithNewOp<VariableGetOp>(callOp, cseVariable);
  }

  return equationTemplateOp;
}

mlir::LogicalResult CallCSEPass::processModelOp(ModelOp modelOp) {
  mlir::IRRewriter rewriter(modelOp);
  mlir::SymbolTableCollection symbolTableCollection;

  llvm::SmallVector<CallOp> callOps;
  collectCallOps(modelOp, callOps);

  llvm::SmallVector<llvm::SmallVector<CallOp>> callEquivalenceGroups;
  buildCallEquivalenceGroups(callOps, symbolTableCollection,
                             callEquivalenceGroups);

  int emittedCSEs = 0;
  llvm::SmallVector<EquationTemplateOp> cseEquationTemplateOps;

  for (auto &equivalenceGroup :
       make_filter_range(callEquivalenceGroups,
                         [](auto &group) { return group.size() > 1; })) {
    cseEquationTemplateOps.push_back(
        emitCse(modelOp, emittedCSEs++, equivalenceGroup, rewriter));
  }

  if (!cseEquationTemplateOps.empty()) {
    rewriter.setInsertionPointToEnd(modelOp.getBody());
    auto dynamicOp = rewriter.create<DynamicOp>(rewriter.getUnknownLoc());
    rewriter.setInsertionPointToStart(
        rewriter.createBlock(&dynamicOp.getRegion()));
    for (auto &equationTemplateOp : cseEquationTemplateOps) {
      rewriter.create<EquationInstanceOp>(rewriter.getUnknownLoc(),
                                          equationTemplateOp);
    }
  }

  return mlir::success();
}

namespace mlir::bmodelica {
std::unique_ptr<Pass> createCallCSEPass() {
  return std::make_unique<CallCSEPass>();
}
} // namespace mlir::bmodelica
