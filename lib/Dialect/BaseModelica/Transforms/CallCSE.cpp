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
};

/// Check if all equation instantiations use the same induction ranges.
bool haveSameInductionRanges(llvm::ArrayRef<EquationInstanceOp> instanceOps) {
  if (instanceOps.size() == 1) {
    return true;
  }
  std::optional<IndexSet> seenSpace;
  for (EquationInstanceOp instanceOp : instanceOps) {
    IndexSet iterationSpace = instanceOp.getIterationSpace();
    if (!seenSpace) {
      seenSpace = iterationSpace;
    } else if (seenSpace.value() != iterationSpace) {
      return false;
    }
  }
  return true;
}

/// Reduces multiple instances to a single template. Completely filters out
/// instances where induction variable ranges do not match. Returns each valid
/// template together with the induction variable ranges it sees.
llvm::SmallVector<
    std::pair<EquationTemplateOp, std::optional<MultidimensionalRange>>>
getTemplates(llvm::SmallVectorImpl<EquationInstanceOp> &instanceOps) {
  // Group instances by template.
  llvm::DenseMap<EquationTemplateOp, llvm::SmallVector<EquationInstanceOp>>
      instancesByTemplate;
  for (EquationInstanceOp equationOp : instanceOps) {
    llvm::SmallVector<EquationInstanceOp> &group =
        instancesByTemplate.getOrInsertDefault(equationOp.getTemplate());
    group.push_back(equationOp);
  }
  // Filter out conflicting instances.
  llvm::SmallVector<
      std::pair<EquationTemplateOp, std::optional<MultidimensionalRange>>>
      filteredTemplates;
  for (auto [templateOp, instanceOps] : instancesByTemplate) {
    if (haveSameInductionRanges(instanceOps)) {
      std::optional<MultidimensionalRange> indices;
      if (auto indicesAttr = instanceOps.front().getIndices()) {
        indices = indicesAttr.value().getValue();
      }
      filteredTemplates.push_back({templateOp, indices});
    }
  }
  // Order templates by their order in the model, to keep result deterministic.
  llvm::sort(filteredTemplates, [](const auto &a, const auto &b) {
    return a.first->isBeforeInBlock(b.first);
  });
  return filteredTemplates;
}

/// Get all call operations in the model with their associated iteration spaces.
llvm::SmallVector<std::pair<CallOp, std::optional<MultidimensionalRange>>>
collectCallOps(ModelOp modelOp) {
  llvm::SmallVector<EquationInstanceOp> dynamicEquationOps;
  modelOp.collectMainEquations(dynamicEquationOps);

  llvm::SmallVector<std::pair<CallOp, std::optional<MultidimensionalRange>>>
      callOpsWithInductionRanges;
  for (auto equationTemplateWithRanges : getTemplates(dynamicEquationOps)) {
    equationTemplateWithRanges.first->walk([&](CallOp callOp) {
      // Skip functions with tensor results
      if (llvm::any_of(callOp.getResultTypes(), [](mlir::Type type) {
            return type.isa<mlir::TensorType>();
          })) {
        return;
      }
      callOpsWithInductionRanges.push_back(
          {callOp, equationTemplateWithRanges.second});
    });
  }
  return callOpsWithInductionRanges;
}

/// A member of a call equivalence group.
///
/// [callOp] is the call operation.
///
/// [templateInductionRanges] are the induction variable *ranges* the template
/// containing [callOp] was instantiated with, if any.
///
/// [orderedUsedInductions] are the induction variables accessed by [callOp],
/// ordered the same as the accesses by the group representative.
struct CallEquivalenceGroupEntry {
  CallOp callOp;
  std::optional<MultidimensionalRange> templateInductionRanges;
  llvm::SmallVector<mlir::BlockArgument> orderedUsedInductions;
};

using CallEquivalenceGroup = llvm::SmallVector<CallEquivalenceGroupEntry>;

/// Partition the list of call operations into groups given by
/// EquationExpressionOpInterface::isEquivalent.
/// The first entry of each group is the group representative, and should be
/// used to produce the cse value. The groups are guaranteed to be non-empty.
llvm::SmallVector<CallEquivalenceGroup> buildEquivalenceGroups(
    const llvm::SmallVectorImpl<
        std::pair<CallOp, std::optional<MultidimensionalRange>>>
        &callOpsWithInductionRanges,
    mlir::SymbolTableCollection &symbolTableCollection) {
  llvm::SmallVector<CallEquivalenceGroup> callEquivalenceGroups;
  for (auto callOpWithInductionRanges : callOpsWithInductionRanges) {
    // Search the existing groups to see if the current call should belong to
    // one of them.
    bool foundGroup = false;
    auto callExpression = mlir::dyn_cast<EquationExpressionOpInterface>(
        *callOpWithInductionRanges.first);
    for (CallEquivalenceGroup &callEquivalenceGroup : callEquivalenceGroups) {
      assert(!callEquivalenceGroup.empty() && "groups cannot not be empty");
      CallEquivalenceGroupEntry representative = callEquivalenceGroup.front();

      // This vector is populated with the 1-1 mapping of induction
      // accesses made by the current call and the representative.
      // The first entry in each pair belongs to the current call.
      llvm::SmallVector<std::pair<mlir::BlockArgument, mlir::BlockArgument>>
          pairedInductions;
      if (!callExpression.isEquivalent(representative.callOp, pairedInductions,
                                       symbolTableCollection)) {
        continue;
      }

      // If no inductions were accessed the calls are equivalent.
      if (pairedInductions.empty()) {
        CallEquivalenceGroupEntry callEntry = {
            .callOp = callOpWithInductionRanges.first,
            .templateInductionRanges = callOpWithInductionRanges.second,
            .orderedUsedInductions = {}};
        callEquivalenceGroup.push_back(callEntry);
        foundGroup = true;
        break;
      }

      // Check that the accessed inductions have the same ranges.
      assert(callOpWithInductionRanges.second &&
             representative.templateInductionRanges &&
             "if inductions are accessed by both operations, they must both "
             "have induction ranges");
      MultidimensionalRange callRanges =
          callOpWithInductionRanges.second.value();
      MultidimensionalRange representativeRanges =
          representative.templateInductionRanges.value();
      bool inductionRangesMatch =
          !llvm::any_of(pairedInductions, [&](auto accessPair) {
            Range callRange = callRanges[accessPair.first.getArgNumber()];
            Range representativeRange =
                representativeRanges[accessPair.second.getArgNumber()];
            return callRange != representativeRange;
          });
      if (!inductionRangesMatch) {
        continue;
      }

      // Reorder induction accesses according to match the representative order.
      // This is so we don't have to rely on the traversal order of
      // EquationExpressionOpInterface::isEquivalent to stay constant.
      llvm::SmallVector<mlir::BlockArgument> callUsedInductions;
      for (mlir::BlockArgument representativeInductionAccess :
           representative.orderedUsedInductions) {
        for (auto accessPair : pairedInductions) {
          if (accessPair.second == representativeInductionAccess) {
            callUsedInductions.push_back(accessPair.first);
            break;
          }
        }
      }

      CallEquivalenceGroupEntry callEntry = {
          .callOp = callOpWithInductionRanges.first,
          .templateInductionRanges = callOpWithInductionRanges.second,
          .orderedUsedInductions = callUsedInductions};
      callEquivalenceGroup.push_back(callEntry);
      foundGroup = true;
      break;
    }

    // If no group was found, create a new one, with the current call as the
    // representative.
    if (!foundGroup) {
      llvm::SmallVector<std::pair<mlir::BlockArgument, mlir::BlockArgument>>
          pairedInductions;
      // Comparing the call to itself is a little hack to get an initial
      // induction access order. This order is arbitrary, and we could traverse
      // the operands ourselves, but this is easier.
      llvm::dyn_cast<EquationExpressionOpInterface>(
          *callOpWithInductionRanges.first)
          .isEquivalent(callOpWithInductionRanges.first, pairedInductions,
                        symbolTableCollection);

      CallEquivalenceGroupEntry representativeEntry = {
          .callOp = callOpWithInductionRanges.first,
          .templateInductionRanges = callOpWithInductionRanges.second,
          .orderedUsedInductions =
              llvm::map_to_vector(pairedInductions, [](auto &accessPair) {
                return accessPair.first;
              })};
      callEquivalenceGroups.push_back({representativeEntry});
    }
  }
  return callEquivalenceGroups;
}

/// For each induction in [inductionsToShift], emit a remapped value so that for
/// an original range a:b, the new value will be 0:(b-a).
///
/// [inductionsToShift] and [originalUsedInductions] are ordered the same way,
/// The original induction ranges are found by indexing into
/// [originalInductionRanges] with the original induction argument number.
llvm::SmallVector<mlir::Value> shiftInductionRanges(
    mlir::ValueRange inductionsToShift,
    const llvm::SmallVectorImpl<mlir::BlockArgument> &originalUsedInductions,
    const MultidimensionalRange &originalInductionRanges,
    mlir::RewriterBase &rewriter, mlir::Location loc) {
  llvm::SmallVector<mlir::Value> shiftedIndices;
  for (auto [induction, originalInduction] :
       llvm::zip(inductionsToShift, originalUsedInductions)) {
    int64_t rangeStart =
        originalInductionRanges[originalInduction.getArgNumber()].getBegin();
    auto offset =
        rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(-rangeStart));
    shiftedIndices.push_back(rewriter.create<AddOp>(loc, induction, offset));
  }
  return shiftedIndices;
}

/// Clone `op` and its def-use chain, returning the cloned version of `op`.
mlir::Operation *cloneDefUseChain(mlir::Operation *op,
                                  mlir::IRMapping &inductionMapping,
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
          if (mlir::Operation *definingOp = operand.getDefiningOp();
              definingOp && definingOp->getBlock() == current->getBlock()) {
            worklist.push_back(definingOp);
          }
        }
      });
    }
  }

  mlir::Operation *root = nullptr;
  for (mlir::Operation *opToClone : llvm::reverse(toClone)) {
    // Skip repeated dependencies on the same operation
    if (inductionMapping.contains(opToClone)) {
      continue;
    }
    root = rewriter.clone(*opToClone, inductionMapping);
  }
  return root;
}

/// Build the template needed to drive [cseVariable] with the result of
/// [representative].
EquationTemplateOp
emitDriverEquationTemplate(const CallEquivalenceGroupEntry &representative,
                           VariableOp cseVariable, const size_t resultIndex,
                           mlir::RewriterBase &rewriter, mlir::Location loc) {
  auto cseEquationTemplateOp = rewriter.create<EquationTemplateOp>(loc);
  // Create the body of the template with |inductions| = the number of
  // inductions used by the representative.
  size_t numInductions = representative.orderedUsedInductions.size();
  rewriter.setInsertionPointToStart(
      cseEquationTemplateOp.createBody(numInductions));

  // Build equation left hand side by getting the cse variable reference.
  auto cseVarRef = rewriter.create<VariableGetOp>(loc, cseVariable);
  EquationSideOp lhsOp;
  if (numInductions == 0) {
    lhsOp = rewriter.create<EquationSideOp>(loc, cseVarRef->getResults());
  } else {
    // As the the induction ranges are on the form a:b, but tensors are
    // 0-based, we need to shift them to start from 0.
    llvm::SmallVector<mlir::Value> shiftedInductionVariables =
        shiftInductionRanges(cseEquationTemplateOp.getInductionVariables(),
                             representative.orderedUsedInductions,
                             representative.templateInductionRanges.value(),
                             rewriter, loc);
    lhsOp = rewriter.create<EquationSideOp>(
        loc,
        rewriter
            .create<TensorExtractOp>(loc, cseVarRef, shiftedInductionVariables)
            ->getResults());
  }

  // As cseTemplateInductions were created from inductionVariableOrder.size() we
  // know these sized match.
  mlir::IRMapping inductionMapping;
  for (auto [originalInduction, cseInduction] :
       llvm::zip(representative.orderedUsedInductions,
                 cseEquationTemplateOp.getInductionVariables())) {
    inductionMapping.map(originalInduction, cseInduction);
  }

  // Assign the result of the call to the cse variable.
  mlir::Operation *clonedRepresentative =
      cloneDefUseChain(representative.callOp, inductionMapping, rewriter);
  auto rhsOp = rewriter.create<EquationSideOp>(
      loc, clonedRepresentative->getResult(resultIndex));
  rewriter.create<EquationSidesOp>(loc, lhsOp, rhsOp);

  return cseEquationTemplateOp;
}

/// Replace all calls in an equivalence group with accesses to the cse
/// variable(s). If the variables are tensors they are indexed with inductions
/// in the order of induction variables present in each entry.
void replaceCalls(const llvm::SmallVectorImpl<CallEquivalenceGroupEntry> &calls,
                  const llvm::SmallVectorImpl<VariableOp> &cseVariables,
                  mlir::RewriterBase &rewriter, mlir::Location loc) {
  for (const CallEquivalenceGroupEntry &call : calls) {
    rewriter.setInsertionPoint(call.callOp);
    llvm::SmallVector<mlir::Value> results =
        llvm::map_to_vector(cseVariables, [&](VariableOp cseVariable) {
          auto cseVarRef = rewriter.create<VariableGetOp>(loc, cseVariable);
          if (call.orderedUsedInductions.empty()) {
            return cseVarRef.getResult();
          }

          auto inductionVariables = llvm::map_to_vector(
              call.orderedUsedInductions, [](mlir::BlockArgument inductionArg) {
                return llvm::dyn_cast<mlir::Value>(inductionArg);
              });
          // As the the induction ranges are on the form a:b, but tensors are
          // 0-based, we need to shift them to start from 0.
          auto shiftedInductions = shiftInductionRanges(
              inductionVariables, call.orderedUsedInductions,
              call.templateInductionRanges.value(), rewriter, loc);
          return rewriter
              .create<TensorExtractOp>(loc, cseVarRef, shiftedInductions)
              .getResult();
        });
    rewriter.replaceOp(call.callOp, results);
  }
}

/// Emit a variable operation with inner type [type], and shape according to
/// [shape]. If [shape] has no value the variable becomes a scalar.
VariableOp emitCseVariable(mlir::Type type,
                           const std::optional<MultidimensionalRange> &shape,
                           mlir::RewriterBase &rewriter, mlir::Location loc) {
  // Build variable shape
  llvm::SmallVector<int64_t> variableShape;
  if (shape) {
    llvm::SmallVector<size_t> inductionSizes;
    shape.value().getSizes(inductionSizes);
    variableShape = llvm::map_to_vector(
        inductionSizes, [](size_t size) { return static_cast<int64_t>(size); });
  }
  return rewriter.create<VariableOp>(
      loc, "_cse", VariableType::wrap(type).withShape(variableShape));
}

/// One variable and driver equation will be emitted per result of the
/// representative's callee, if the call is to a function with multiple result
/// values.
llvm::SmallVector<VariableOp> emitCseVariables(
    CallEquivalenceGroupEntry representative,
    std::optional<MultidimensionalRange> &cseTemplateInductionRanges,
    ModelOp modelOp, DynamicOp dynamicOp, mlir::SymbolTable &symbolTable,
    mlir::RewriterBase &rewriter, mlir::Location loc) {
  // Build cse variable iteration space
  if (representative.templateInductionRanges) {
    for (mlir::BlockArgument inductionVariable :
         representative.orderedUsedInductions) {
      auto inductionRange =
          MultidimensionalRange(representative.templateInductionRanges
                                    .value()[inductionVariable.getArgNumber()]);
      if (cseTemplateInductionRanges) {
        cseTemplateInductionRanges =
            cseTemplateInductionRanges.value().append(inductionRange);
      } else {
        cseTemplateInductionRanges = inductionRange;
      }
    }
  }

  // Emit one variable per function result
  llvm::SmallVector<VariableOp> cseVariables;
  for (auto result : llvm::enumerate(representative.callOp.getResults())) {
    rewriter.setInsertionPointToStart(modelOp.getBody());

    // Emit cse variable
    auto cseVariable = emitCseVariable(
        result.value().getType(), cseTemplateInductionRanges, rewriter, loc);
    symbolTable.insert(cseVariable);
    cseVariables.push_back(cseVariable);

    // Emit driver equation
    rewriter.setInsertionPoint(dynamicOp);
    EquationTemplateOp driverTemplate = emitDriverEquationTemplate(
        representative, cseVariable, result.index(), rewriter, loc);

    // Instantiate driver equation (with indices if needed).
    rewriter.setInsertionPointToEnd(dynamicOp.getBody());
    if (!cseTemplateInductionRanges) {
      rewriter.create<EquationInstanceOp>(loc, driverTemplate);
    } else {
      auto indices = MultidimensionalRangeAttr::get(
          rewriter.getContext(), cseTemplateInductionRanges.value());
      rewriter.create<EquationInstanceOp>(loc, driverTemplate, indices);
    }
  }

  return cseVariables;
}

/// Perform common subexpression elimination on the model.
mlir::LogicalResult CallCSEPass::processModelOp(ModelOp modelOp) {
  mlir::IRRewriter rewriter(modelOp);
  mlir::SymbolTableCollection symbolTableCollection;

  llvm::SmallVector<CallEquivalenceGroup> callEquivalenceGroups =
      buildEquivalenceGroups(collectCallOps(modelOp), symbolTableCollection);

  // Remove singleton groups
  llvm::erase_if(callEquivalenceGroups, [](const CallEquivalenceGroup &group) {
    return group.size() < 2;
  });
  if (callEquivalenceGroups.empty()) {
    return mlir::success();
  }

  mlir::SymbolTable symbolTable(modelOp);
  rewriter.setInsertionPointToEnd(modelOp.getBody());
  DynamicOp dynamicOp = rewriter.create<DynamicOp>(rewriter.getUnknownLoc());
  rewriter.createBlock(&dynamicOp.getBodyRegion());

  for (CallEquivalenceGroup &equivalenceGroup : callEquivalenceGroups) {
    CallEquivalenceGroupEntry representative = equivalenceGroup.front();
    // Only emit CSEs that will lead to an equivalent, or lower amount of calls.
    // This because calls to functions with n results will generate n driver
    // equations.
    if (equivalenceGroup.size() < representative.callOp.getNumResults()) {
      continue;
    }

    mlir::Location loc = representative.callOp.getLoc();
    std::optional<MultidimensionalRange> cseInductionRanges;
    llvm::SmallVector<VariableOp> cseVariables =
        emitCseVariables(representative, cseInductionRanges, modelOp, dynamicOp,
                         symbolTable, rewriter, loc);

    replaceCalls(equivalenceGroup, cseVariables, rewriter, loc);

    ++this->newCSEVariables;
    int replacedCalls = equivalenceGroup.size();
    if (cseInductionRanges) {
      replacedCalls *= cseInductionRanges.value().flatSize();
    }
    this->replacedCalls += replacedCalls;
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