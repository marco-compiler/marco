#include "marco/Dialect/BaseModelica/Transforms/EquationAccessSplit.h"
#include "marco/Dialect/BaseModelica/Analysis/VariableAccessAnalysis.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_EQUATIONACCESSSPLITPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class EquationAccessSplitPass
    : public impl::EquationAccessSplitPassBase<EquationAccessSplitPass>,
      public VariableAccessAnalysis::AnalysisProvider {
public:
  using EquationAccessSplitPassBase<
      EquationAccessSplitPass>::EquationAccessSplitPassBase;

  void runOnOperation() override;

  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getCachedVariableAccessAnalysis(EquationTemplateOp op) override;

private:
  mlir::LogicalResult processModelOp(ModelOp modelOp);
};
} // namespace

void EquationAccessSplitPass::runOnOperation() {
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

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
EquationAccessSplitPass::getCachedVariableAccessAnalysis(
    EquationTemplateOp op) {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::Operation *parentOp = op->getParentOp();
  llvm::SmallVector<mlir::Operation *> parentOps;

  while (parentOp != moduleOp) {
    parentOps.push_back(parentOp);
    parentOp = parentOp->getParentOp();
  }

  mlir::AnalysisManager analysisManager = getAnalysisManager();

  for (mlir::Operation *currentParentOp : llvm::reverse(parentOps)) {
    analysisManager = analysisManager.nest(currentParentOp);
  }

  return analysisManager.getCachedChildAnalysis<VariableAccessAnalysis>(op);
}

namespace {
void collectEquations(llvm::SmallVectorImpl<EquationInstanceOp> &equationOps,
                      mlir::Region &region) {
  llvm::append_range(equationOps, region.getOps<EquationInstanceOp>());
}

struct Partition {
  IndexSet equationIndices;
  IndexSet variableIndices;
  llvm::SmallVector<VariableAccess> accesses;
};

mlir::LogicalResult
splitEquationIndices(mlir::RewriterBase &rewriter,
                     mlir::SymbolTableCollection &symbolTableCollection,
                     EquationInstanceOp op) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);

  if (op.getProperties().indices.empty()) {
    return mlir::success();
  }

  const Variable &matchedVariable = op.getProperties().match;
  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed(op.getAccesses(accesses, symbolTableCollection))) {
    return mlir::failure();
  }

  llvm::SmallVector<Partition> variablePartitions;
  variablePartitions.push_back({.variableIndices = matchedVariable.indices});

  llvm::SmallVector<VariableAccess> matchAccesses;

  for (const VariableAccess &access : accesses) {
    const AccessFunction &accessFunction = access.getAccessFunction();

    if (access.getVariable() != matchedVariable.name) {
      continue;
    }

    IndexSet accessedIndices = accessFunction.map(op.getProperties().indices);

    if (!matchedVariable.indices.contains(accessedIndices)) {
      continue;
    }

    matchAccesses.push_back(access);
    llvm::SmallVector<Partition> newVariablePartitions;

    for (Partition &partition : variablePartitions) {
      IndexSet intersection =
          accessedIndices.intersect(partition.variableIndices);

      if (intersection.empty()) {
        newVariablePartitions.push_back(std::move(partition));
      } else {
        Partition &intersectionPartition = newVariablePartitions.emplace_back();
        intersectionPartition.variableIndices = intersection;
        intersectionPartition.accesses = partition.accesses;
        intersectionPartition.accesses.push_back(access);

        if (IndexSet diff = partition.variableIndices - intersection;
            !diff.empty()) {
          Partition &diffPartition = newVariablePartitions.emplace_back();
          diffPartition.variableIndices = diff;
          diffPartition.accesses = partition.accesses;
        }
      }
    }

    variablePartitions = std::move(newVariablePartitions);
  }

  if (matchAccesses.size() <= 1) {
    return mlir::success();
  }

  llvm::SmallVector<Partition> equationPartitions;
  equationPartitions.push_back({.equationIndices = op.getProperties().indices});

  for (Partition &variablePartition : variablePartitions) {
    llvm::SmallVector<Partition> newEquationPartitions;

    IndexSet inverseMap = op.getProperties().indices;

    for (const VariableAccess &access : variablePartition.accesses) {
      const AccessFunction &accessFunction = access.getAccessFunction();

      inverseMap = inverseMap.intersect(accessFunction.inverseMap(
          variablePartition.variableIndices, op.getProperties().indices));
    }

    for (Partition &equationPartition : equationPartitions) {
      IndexSet intersection =
          inverseMap.intersect(equationPartition.equationIndices);

      if (intersection.empty()) {
        newEquationPartitions.push_back(std::move(equationPartition));
      } else {
        Partition &intersectionPartition = newEquationPartitions.emplace_back();
        intersectionPartition.variableIndices = {};
        intersectionPartition.equationIndices = intersection;
        intersectionPartition.accesses = equationPartition.accesses;

        llvm::append_range(intersectionPartition.accesses,
                           variablePartition.accesses);

        for (const VariableAccess &access : intersectionPartition.accesses) {
          intersectionPartition.variableIndices +=
              access.getAccessFunction().map(
                  intersectionPartition.equationIndices);
        }

        if (IndexSet diff = equationPartition.equationIndices - intersection;
            !diff.empty()) {
          Partition &diffPartition = newEquationPartitions.emplace_back();
          diffPartition.variableIndices = {};
          diffPartition.equationIndices = diff;
          diffPartition.accesses = equationPartition.accesses;

          for (const VariableAccess &access : diffPartition.accesses) {
            diffPartition.variableIndices +=
                access.getAccessFunction().map(diffPartition.equationIndices);
          }
        }
      }
    }

    equationPartitions = std::move(newEquationPartitions);
  }

  // Check that the split equation indices do represent exactly the initial set.
  assert(std::accumulate(equationPartitions.begin(), equationPartitions.end(),
                         IndexSet(),
                         [](const IndexSet &acc, const auto &partition) {
                           return acc + partition.equationIndices;
                         }) == op.getProperties().indices);

  // Check that the split variable indices do represent exactly the initial set.
  assert(std::accumulate(equationPartitions.begin(), equationPartitions.end(),
                         IndexSet(),
                         [](const IndexSet &acc, const auto &partition) {
                           return acc + partition.variableIndices;
                         }) == op.getProperties().match.indices);

  // Check that the equation indices exist exactly once in the partitions.
  assert(llvm::all_of(op.getProperties().indices, [&](Point point) {
    return llvm::count_if(equationPartitions, [&](const auto &partition) {
             return partition.equationIndices.contains(point);
           }) == 1;
  }));

  // Check that the matched variable indices exist exactly once in the
  // partitions.
  assert(llvm::all_of(op.getProperties().match.indices, [&](Point point) {
    return llvm::count_if(equationPartitions, [&](const auto &partition) {
             return partition.variableIndices.contains(point);
           }) == 1;
  }));

  if (equationPartitions.size() == 1) {
    return mlir::success();
  }

  for (Partition &partition : equationPartitions) {
    assert(!partition.equationIndices.empty());
    assert(!partition.variableIndices.empty());
    assert(!partition.accesses.empty());

    auto clonedOp =
        mlir::cast<EquationInstanceOp>(rewriter.clone(*op.getOperation()));

    clonedOp.getProperties().indices = std::move(partition.equationIndices);

    clonedOp.getProperties().match.indices =
        std::move(partition.variableIndices);
  }

  rewriter.eraseOp(op);
  return mlir::success();
}
} // namespace

mlir::LogicalResult EquationAccessSplitPass::processModelOp(ModelOp modelOp) {
  llvm::SmallVector<EquationInstanceOp> equationOps;

  for (mlir::Operation &nestedOp : modelOp.getOps()) {
    if (auto initialOp = mlir::dyn_cast<InitialOp>(nestedOp)) {
      collectEquations(equationOps, initialOp.getBodyRegion());
    }

    if (auto dynamicOp = mlir::dyn_cast<DynamicOp>(nestedOp)) {
      collectEquations(equationOps, dynamicOp.getBodyRegion());
      continue;
    }
  }

  mlir::IRRewriter rewriter(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;

  for (EquationInstanceOp equationOp : equationOps) {
    if (mlir::failed(splitEquationIndices(rewriter, symbolTableCollection,
                                          equationOp))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createEquationAccessSplitPass() {
  return std::make_unique<EquationAccessSplitPass>();
}
} // namespace mlir::bmodelica
