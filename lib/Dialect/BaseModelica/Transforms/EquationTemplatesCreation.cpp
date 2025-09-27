#include "marco/Dialect/BaseModelica/Transforms/EquationTemplatesCreation.h"

#include "marco/AST/BaseModelica/Node/Equation.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_EQUATIONTEMPLATESCREATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class EquationTemplatesCreationPass
    : public impl::EquationTemplatesCreationPassBase<
          EquationTemplatesCreationPass> {
public:
  using EquationTemplatesCreationPassBase<
      EquationTemplatesCreationPass>::EquationTemplatesCreationPassBase;

  void runOnOperation() override;

private:
  void createTemplates(ModelOp modelOp);
};
} // namespace

void EquationTemplatesCreationPass::runOnOperation() {
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  mlir::parallelForEach(&getContext(), modelOps, [&](mlir::Operation *op) {
    createTemplates(mlir::cast<ModelOp>(op));
  });
}

using EquationInfo = std::pair<mlir::Operation *, EquationOp>;
using ForEquationInfo = std::pair<mlir::Operation *, ForEquationOp>;

namespace {
void collectEquationAndForEquationOps(
    mlir::Operation *op, mlir::Operation *dynamicOrInitialOp,
    llvm::SmallVectorImpl<EquationInfo> &equations,
    llvm::SmallVectorImpl<ForEquationInfo> &forEquations) {
  for (mlir::Region &region : op->getRegions()) {
    for (mlir::Operation &nestedOp : region.getOps()) {
      if (auto equationOp = mlir::dyn_cast<EquationOp>(nestedOp)) {
        equations.emplace_back(dynamicOrInitialOp, equationOp);
        continue;
      }

      if (auto forEquationOp = mlir::dyn_cast<ForEquationOp>(nestedOp)) {
        forEquations.emplace_back(dynamicOrInitialOp, forEquationOp);
      }
    }
  }
}

void setInsertionPointToEndOfContainerOp(mlir::OpBuilder &builder,
                                         mlir::Operation *dynamicOrInitialOp) {
  if (auto dynamicOp = mlir::dyn_cast<DynamicOp>(dynamicOrInitialOp)) {
    builder.setInsertionPointToEnd(dynamicOp.getBody());
    return;
  }

  if (auto initialOp = mlir::dyn_cast<InitialOp>(dynamicOrInitialOp)) {
    builder.setInsertionPointToStart(initialOp.getBody());
  }
}

void getForEquationOps(EquationOp op,
                       llvm::SmallVectorImpl<ForEquationOp> &result) {
  llvm::SmallVector<ForEquationOp> loopsStack;
  auto parentLoop = op->getParentOfType<ForEquationOp>();

  while (parentLoop) {
    loopsStack.push_back(parentLoop);
    parentLoop = parentLoop->getParentOfType<ForEquationOp>();
  }

  while (!loopsStack.empty()) {
    result.push_back(loopsStack.pop_back_val());
  }
}

std::vector<MultidimensionalRange> getRangesCombinations(
    size_t rank, llvm::function_ref<llvm::ArrayRef<Range>(size_t)> rangesFn,
    size_t startingDimension) {
  assert(startingDimension < rank);
  std::vector<MultidimensionalRange> result;

  if (startingDimension == rank - 1) {
    for (const Range &range : rangesFn(startingDimension)) {
      result.emplace_back(range);
    }

    return result;
  }

  auto subCombinations =
      getRangesCombinations(rank, rangesFn, startingDimension + 1);

  assert(!rangesFn(startingDimension).empty());

  for (const Range &range : rangesFn(startingDimension)) {
    for (const auto &subCombination : subCombinations) {
      llvm::SmallVector<Range, 1> current;
      current.push_back(range);

      for (size_t i = 0; i < subCombination.rank(); ++i) {
        current.push_back(subCombination[i]);
      }

      result.emplace_back(current);
    }
  }

  return result;
}

std::vector<MultidimensionalRange> getRangesCombinations(
    size_t rank, llvm::function_ref<llvm::ArrayRef<Range>(size_t)> rangesFn) {
  return getRangesCombinations(rank, rangesFn, 0);
}

IndexSet getIterationSpace(llvm::ArrayRef<ForEquationOp> loops) {
  if (loops.empty()) {
    return {};
  }

  llvm::SmallVector<llvm::SmallVector<Range, 1>, 3> dimensionsRanges;
  dimensionsRanges.resize(loops.size());

  size_t forEquationIndex = 0;

  for (ForEquationOp forEquationOp : loops) {
    auto from = forEquationOp.getFrom().getSExtValue();
    auto to = forEquationOp.getTo().getSExtValue();
    auto step = forEquationOp.getStep().getSExtValue();

    if (step == 1) {
      dimensionsRanges[forEquationIndex].emplace_back(from, to + 1);
    } else {
      for (auto index = from; index < to + 1; index += step) {
        dimensionsRanges[forEquationIndex].emplace_back(index, index + 1);
      }
    }

    ++forEquationIndex;
  }

  return IndexSet(getRangesCombinations(
      dimensionsRanges.size(), [&](size_t dimension) -> llvm::ArrayRef<Range> {
        return dimensionsRanges[dimension];
      }));
}
} // namespace

void EquationTemplatesCreationPass::createTemplates(ModelOp modelOp) {
  mlir::IRRewriter rewriter(modelOp);

  llvm::SmallVector<EquationInfo> equations;
  llvm::SmallVector<ForEquationInfo> forEquations;
  llvm::SmallVector<mlir::Operation *> toBeErased;

  for (mlir::Operation &nestedOp : modelOp.getOps()) {
    if (mlir::isa<DynamicOp, InitialOp>(nestedOp)) {
      collectEquationAndForEquationOps(&nestedOp, &nestedOp, equations,
                                       forEquations);
    }
  }

  for (const EquationInfo &equation : equations) {
    toBeErased.push_back(equation.second);
  }

  for (const ForEquationInfo &forEquation : forEquations) {
    toBeErased.push_back(forEquation.second);
  }

  while (!forEquations.empty()) {
    ForEquationInfo current = forEquations.pop_back_val();
    collectEquationAndForEquationOps(current.second, current.first, equations,
                                     forEquations);
  }

  for (const EquationInfo &equation : equations) {
    mlir::Operation *dynamicOrInitialOp = equation.first;
    EquationOp equationOp = equation.second;
    mlir::Location loc = equationOp.getLoc();

    // Create the equation template.
    rewriter.setInsertionPoint(dynamicOrInitialOp);

    llvm::SmallVector<ForEquationOp, 3> loops;
    getForEquationOps(equationOp, loops);

    llvm::SmallVector<mlir::Value, 3> oldInductions;

    for (ForEquationOp forEquationOp : loops) {
      oldInductions.push_back(forEquationOp.induction());
    }

    auto templateOp = rewriter.create<EquationTemplateOp>(loc);
    mlir::Block *templateBody = templateOp.createBody(loops.size());

    mlir::IRMapping mapping;

    for (const auto &[oldInduction, newInduction] :
         llvm::zip(oldInductions, templateOp.getInductionVariables())) {
      mapping.map(oldInduction, newInduction);
    }

    rewriter.setInsertionPointToStart(templateBody);

    for (auto &nestedOp : equationOp.getOps()) {
      rewriter.clone(nestedOp, mapping);
    }

    // Create the equation instances.
    setInsertionPointToEndOfContainerOp(rewriter, dynamicOrInitialOp);

    auto instanceOp = rewriter.create<EquationInstanceOp>(loc, templateOp);

    instanceOp.getProperties().indices = getIterationSpace(loops);
  }

  for (mlir::Operation *op : toBeErased) {
    rewriter.eraseOp(op);
  }
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createEquationTemplatesCreationPass() {
  return std::make_unique<EquationTemplatesCreationPass>();
}
} // namespace mlir::bmodelica
