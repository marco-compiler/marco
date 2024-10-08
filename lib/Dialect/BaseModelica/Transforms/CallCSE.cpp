#include "marco/Dialect/BaseModelica/Transforms/CallCSE.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_CALLCSEPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class CallCSEPass final : public impl::CallCSEPassBase<CallCSEPass> {
public:
  using CallCSEPassBase<CallCSEPass>::CallCSEPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult processModelOp(ModelOp modelOp);

  /// Replace all calls in the equivalence group with gets to a generated
  /// variable. The variable will be driven by an equation derived from the
  /// first call in the group.
  ///
  /// One variable and driver equation will be emitted per result,
  /// if the call is to a function with multiple result values.
  void emitCse(llvm::SmallVectorImpl<CallOp> &equivalenceGroup, ModelOp modelOp,
               DynamicOp dynamicOp, mlir::SymbolTable &symbolTable,
               mlir::RewriterBase &rewriter);
};

/// Get all call operations in the model.
void collectCallOps(ModelOp modelOp, llvm::SmallVectorImpl<CallOp> &callOps) {
  llvm::SmallVector<EquationInstanceOp> dynamicEquationOps;
  modelOp.collectMainEquations(dynamicEquationOps);

  llvm::DenseSet<EquationTemplateOp> visitedTemplateOps;
  for (EquationInstanceOp equationOp : dynamicEquationOps) {
    EquationTemplateOp templateOp = equationOp.getTemplate();
    if (!templateOp.getInductionVariables().empty() ||
        visitedTemplateOps.contains(templateOp)) {
      continue;
    }
    visitedTemplateOps.insert(templateOp);
    templateOp->walk([&](CallOp callOp) { callOps.push_back(callOp); });
  }
}

/// Partition the list of call operations into groups given by
/// EquationExpressionOpInterface::isEquivalent
void buildCallEquivalenceGroups(
    llvm::SmallVectorImpl<CallOp> &callOps,
    llvm::SmallVectorImpl<llvm::SmallVector<CallOp>> &callEquivalenceGroups) {
  mlir::SymbolTableCollection symbolTableCollection;
  llvm::SmallVector<llvm::SmallVector<CallOp>> tmpCallEquivalenceGroups;

  for (CallOp callOp : callOps) {
    auto callExpression =
        mlir::cast<EquationExpressionOpInterface>(callOp.getOperation());

    llvm::SmallVector<CallOp> *equivalenceGroup = find_if(
        tmpCallEquivalenceGroups, [&](llvm::SmallVector<CallOp> &group) {
          assert(!group.empty() && "groups should never be empty");
          return callExpression.isEquivalent(group.front(),
                                             symbolTableCollection);
        });

    if (equivalenceGroup != tmpCallEquivalenceGroups.end()) {
      // Add equivalent call to existing group
      equivalenceGroup->push_back(callOp);
    } else {
      // Create new equivalence group
      tmpCallEquivalenceGroups.push_back({callOp});
    }
  }

  for (llvm::SmallVector<CallOp> &group : tmpCallEquivalenceGroups) {
    if (group.size() > 1) {
      callEquivalenceGroups.push_back(std::move(group));
    }
  }
}

/// Clone `op` and its def-use chain, returning the cloned version of `op`.
mlir::Operation *cloneDefUseChain(mlir::Operation *op,
                                  mlir::RewriterBase &rewriter) {
  llvm::SmallVector<mlir::Operation *> toClone;
  llvm::SmallVector<mlir::Operation *> worklist({op});

  // DFS through the def-use chain of `op`
  while (!worklist.empty()) {
    mlir::Operation *current = worklist.back();
    worklist.pop_back();
    toClone.push_back(current);
    for (mlir::Value operand : current->getOperands()) {
      if (mlir::Operation *defOp = operand.getDefiningOp()) {
        worklist.push_back(defOp);
      }
    }
    // Find the dependencies on operations not defined within the regions of
    // `current`. No need to do this if it is isolated from above.
    if (!current->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
      // Find all uses of values defined outside `current`.
      current->walk([&](mlir::Operation *childOp) {
        // Walk includes current, so skip it.
        if (childOp == current) {
          return;
        }
        for (mlir::Value operand : childOp->getOperands()) {
          // If an operand is defined in the same scope as `current`,
          // i.e. the equation template scope, add it to the worklist.
          mlir::Operation *definingOp = operand.getDefiningOp();
          if (definingOp && definingOp->getBlock() == current->getBlock()) {
            worklist.push_back(definingOp);
          }
        }
      });
    }
  }

  mlir::IRMapping mapping;
  mlir::Operation *root = nullptr;
  for (mlir::Operation *opToClone : llvm::reverse(toClone)) {
    // Skip repeated dependencies on the same operation
    if (mapping.contains(opToClone)) {
      continue;
    }
    root = rewriter.clone(*opToClone, mapping);
  }
  return root;
}

void CallCSEPass::emitCse(llvm::SmallVectorImpl<CallOp> &equivalenceGroup,
                          ModelOp modelOp, DynamicOp dynamicOp,
                          mlir::SymbolTable &symbolTable,
                          mlir::RewriterBase &rewriter) {
  assert(!equivalenceGroup.empty() && "equivalenceGroup cannot be empty");
  CallOp representative = equivalenceGroup.front();
  const mlir::Location loc = representative.getLoc();

  // Emit one variable per function result
  llvm::SmallVector<VariableOp> cseVariables;
  for (auto result : llvm::enumerate(representative.getResults())) {
    rewriter.setInsertionPointToStart(modelOp.getBody());
    // Emit cse variable
    auto cseVariable = rewriter.create<VariableOp>(
        loc, "_cse", VariableType::wrap(result.value().getType()));
    symbolTable.insert(cseVariable);
    cseVariables.push_back(cseVariable);

    // Emit driver equation
    rewriter.setInsertionPoint(dynamicOp);
    auto equationTemplateOp = rewriter.create<EquationTemplateOp>(loc);
    rewriter.setInsertionPointToStart(equationTemplateOp.createBody(0));
    auto lhsOp = rewriter.create<EquationSideOp>(
        loc, rewriter.create<VariableGetOp>(loc, cseVariable)->getResults());
    auto rhsOp = rewriter.create<EquationSideOp>(
        loc,
        cloneDefUseChain(representative, rewriter)->getResult(result.index()));
    rewriter.create<EquationSidesOp>(loc, lhsOp, rhsOp);

    // Add driver equation to dynamic operation
    rewriter.setInsertionPointToEnd(dynamicOp.getBody());
    rewriter.create<EquationInstanceOp>(rewriter.getUnknownLoc(),
                                        equationTemplateOp);
  }

  // Replace calls with get(s) to CSE variable(s)
  for (auto &callOp : equivalenceGroup) {
    rewriter.setInsertionPoint(callOp);

    llvm::SmallVector<mlir::Value> results;
    for (VariableOp cseVariable : cseVariables) {
      results.push_back(
          rewriter.create<VariableGetOp>(loc, cseVariable).getResult());
    }
    rewriter.replaceOp(callOp, results);
  }

  this->replacedCalls += equivalenceGroup.size();
  ++this->newCSEVariables;
}

mlir::LogicalResult CallCSEPass::processModelOp(ModelOp modelOp) {
  mlir::IRRewriter rewriter(modelOp);
  mlir::SymbolTable symbolTable(modelOp);

  llvm::SmallVector<CallOp> callOps;
  collectCallOps(modelOp, callOps);

  llvm::SmallVector<llvm::SmallVector<CallOp>> callEquivalenceGroups;
  buildCallEquivalenceGroups(callOps, callEquivalenceGroups);

  if (callEquivalenceGroups.empty()) {
    return mlir::success();
  }

  rewriter.setInsertionPointToEnd(modelOp.getBody());
  DynamicOp dynamicOp = rewriter.create<DynamicOp>(rewriter.getUnknownLoc());
  rewriter.createBlock(&dynamicOp.getRegion());

  for (llvm::SmallVector<CallOp> &equivalenceGroup : callEquivalenceGroups) {
    // Only emit CSEs that will lead to an equivalent, or lower amount of calls
    if (equivalenceGroup.size() >= equivalenceGroup.front().getNumResults()) {
      emitCse(equivalenceGroup, modelOp, dynamicOp, symbolTable, rewriter);
    }
  }

  if (dynamicOp.getBody()->empty()) {
    rewriter.eraseOp(dynamicOp);
  }

  return mlir::success();
}
} // namespace

void CallCSEPass::runOnOperation() {
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), modelOps, [&](mlir::Operation *op) {
            return processModelOp(mlir::cast<ModelOp>(op));
          }))) {
    return signalPassFailure();
  }
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createCallCSEPass() {
  return std::make_unique<CallCSEPass>();
}
} // namespace mlir::bmodelica