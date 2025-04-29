#include "marco/Dialect/BaseModelica/Transforms/ScheduleParallelization.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_SCHEDULEPARALLELIZATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class ScheduleParallelizationPass
    : public mlir::bmodelica::impl::ScheduleParallelizationPassBase<
          ScheduleParallelizationPass> {
public:
  static const int64_t kUnlimitedGroupBlocks = -1;

public:
  using ScheduleParallelizationPassBase<
      ScheduleParallelizationPass>::ScheduleParallelizationPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult processModelOp(ModelOp modelOp);

  mlir::LogicalResult
  processScheduleOp(mlir::SymbolTableCollection &symbolTableCollection,
                    ModelOp modelOp, ScheduleOp scheduleOp);

  mlir::LogicalResult
  processInitialOps(mlir::SymbolTableCollection &symbolTableCollection,
                    ModelOp modelOp, llvm::ArrayRef<InitialOp> initialOps);

  mlir::LogicalResult
  processInitialOp(mlir::SymbolTableCollection &symbolTableCollection,
                   ModelOp modelOp, InitialOp initialOp);

  mlir::LogicalResult
  processDynamicOps(mlir::SymbolTableCollection &symbolTableCollection,
                    ModelOp modelOp, llvm::ArrayRef<DynamicOp> dynamicOps);

  mlir::LogicalResult
  processDynamicOp(mlir::SymbolTableCollection &symbolTableCollection,
                   ModelOp modelOp, DynamicOp dynamicOp);

  mlir::LogicalResult
  parallelizeBlocks(mlir::SymbolTableCollection &symbolTableCollection,
                    ModelOp modelOp, llvm::ArrayRef<ScheduleBlockOp> blocks);
};
} // namespace

void ScheduleParallelizationPass::runOnOperation() {
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

mlir::LogicalResult
ScheduleParallelizationPass::processModelOp(ModelOp modelOp) {
  mlir::SymbolTableCollection symbolTableCollection;

  for (ScheduleOp scheduleOp : modelOp.getOps<ScheduleOp>()) {
    if (mlir::failed(
            processScheduleOp(symbolTableCollection, modelOp, scheduleOp))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult ScheduleParallelizationPass::processScheduleOp(
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    ScheduleOp scheduleOp) {
  llvm::SmallVector<InitialOp> initialOps;
  llvm::SmallVector<DynamicOp> dynamicOps;

  for (auto &op : scheduleOp.getOps()) {
    if (auto initialOp = mlir::dyn_cast<InitialOp>(op)) {
      initialOps.push_back(initialOp);
      continue;
    }

    if (auto dynamicOp = mlir::dyn_cast<DynamicOp>(op)) {
      dynamicOps.push_back(dynamicOp);
      continue;
    }
  }

  if (mlir::failed(
          processInitialOps(symbolTableCollection, modelOp, initialOps))) {
    return mlir::failure();
  }

  if (mlir::failed(
          processDynamicOps(symbolTableCollection, modelOp, dynamicOps))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult ScheduleParallelizationPass::processInitialOps(
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    llvm::ArrayRef<InitialOp> initialOps) {
  for (InitialOp initialOp : initialOps) {
    if (mlir::failed(
            processInitialOp(symbolTableCollection, modelOp, initialOp))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult ScheduleParallelizationPass::processInitialOp(
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    InitialOp initialOp) {
  llvm::SmallVector<ScheduleBlockOp> blocks;

  for (auto &op : llvm::make_early_inc_range(initialOp.getOps())) {
    if (auto blockOp = mlir::dyn_cast<ScheduleBlockOp>(op)) {
      if (blockOp.getParallelizable()) {
        blocks.push_back(blockOp);
      } else {
        if (mlir::failed(
                parallelizeBlocks(symbolTableCollection, modelOp, blocks))) {
          return mlir::failure();
        }

        blocks.clear();
      }
    } else {
      if (mlir::failed(
              parallelizeBlocks(symbolTableCollection, modelOp, blocks))) {
        return mlir::failure();
      }

      blocks.clear();
    }
  }

  // Parallelize the last chunk of blocks.
  return parallelizeBlocks(symbolTableCollection, modelOp, blocks);
}

mlir::LogicalResult ScheduleParallelizationPass::processDynamicOps(
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    llvm::ArrayRef<DynamicOp> dynamicOps) {
  for (DynamicOp dynamicOp : dynamicOps) {
    if (mlir::failed(
            processDynamicOp(symbolTableCollection, modelOp, dynamicOp))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult ScheduleParallelizationPass::processDynamicOp(
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    DynamicOp dynamicOp) {
  llvm::SmallVector<ScheduleBlockOp> blocks;

  for (auto &op : llvm::make_early_inc_range(dynamicOp.getOps())) {
    if (auto blockOp = mlir::dyn_cast<ScheduleBlockOp>(op)) {
      if (blockOp.getParallelizable()) {
        blocks.push_back(blockOp);
      } else {
        if (mlir::failed(
                parallelizeBlocks(symbolTableCollection, modelOp, blocks))) {
          return mlir::failure();
        }

        blocks.clear();
      }
    } else {
      if (mlir::failed(
              parallelizeBlocks(symbolTableCollection, modelOp, blocks))) {
        return mlir::failure();
      }

      blocks.clear();
    }
  }

  // Parallelize the last chunk of blocks.
  return parallelizeBlocks(symbolTableCollection, modelOp, blocks);
}

mlir::LogicalResult ScheduleParallelizationPass::parallelizeBlocks(
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    llvm::ArrayRef<ScheduleBlockOp> blocks) {
  if (blocks.empty()) {
    return mlir::success();
  }

  // Compute the writes map.
  WritesMap<VariableOp, ScheduleBlockOp> writesMap;

  if (mlir::failed(
          getWritesMap(writesMap, modelOp, blocks, symbolTableCollection))) {
    return mlir::failure();
  }

  // Compute the outgoing arcs and the in-degree of each block.
  using Dependencies = llvm::SetVector<ScheduleBlockOp>;
  llvm::DenseMap<ScheduleBlockOp, Dependencies> dependantBlocks;
  llvm::DenseMap<ScheduleBlockOp, size_t> inDegrees;

  for (ScheduleBlockOp block : blocks) {
    inDegrees[block] = 0;
  }

  for (ScheduleBlockOp readingBlock : blocks) {
    for (const Variable &readVariable :
         readingBlock.getProperties().readVariables) {
      auto readVariableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
          modelOp, readVariable.name);

      for (const auto &writingBlock :
           writesMap.getWrites(readVariableOp, readVariable.indices)) {
        if (writingBlock.writingEntity == readingBlock) {
          // Ignore self-loops.
          continue;
        }

        dependantBlocks[writingBlock.writingEntity].insert(readingBlock);
        inDegrees[readingBlock]++;
      }
    }
  }

  // Compute the sets of independent blocks.
  llvm::SmallVector<llvm::SmallVector<ScheduleBlockOp, 32>, 10> groups;
  llvm::SmallVector<ScheduleBlockOp> currentBlocks;

  for (ScheduleBlockOp block : blocks) {
    if (inDegrees[block] == 0) {
      currentBlocks.push_back(block);
    }
  }

  while (!currentBlocks.empty()) {
    llvm::SmallVector<ScheduleBlockOp> newBlocks;
    llvm::SmallVector<ScheduleBlockOp, 32> independentBlocks;

    for (ScheduleBlockOp block : currentBlocks) {
      assert(inDegrees[block] == 0);

      if (maxParallelBlocks == kUnlimitedGroupBlocks ||
          static_cast<int64_t>(independentBlocks.size()) < maxParallelBlocks) {
        independentBlocks.push_back(block);

        for (ScheduleBlockOp dependantBlock : dependantBlocks[block]) {
          assert(inDegrees[dependantBlock] > 0);

          if (--inDegrees[dependantBlock] == 0) {
            newBlocks.push_back(dependantBlock);
          }
        }
      } else {
        newBlocks.push_back(block);
      }
    }

    assert(!independentBlocks.empty());
    groups.push_back(std::move(independentBlocks));

    currentBlocks = std::move(newBlocks);
    newBlocks.clear();
  }

  // Create the operation containing the parallel blocks.
  mlir::IRRewriter rewriter(&getContext());
  rewriter.setInsertionPointAfter(blocks.back());

  for (const auto &group : groups) {
    auto parallelBlocksOp =
        rewriter.create<ParallelScheduleBlocksOp>(modelOp.getLoc());

    mlir::OpBuilder::InsertionGuard guard(rewriter);

    mlir::Block *bodyBlock =
        rewriter.createBlock(&parallelBlocksOp.getBodyRegion());

    for (ScheduleBlockOp block : group) {
      block.getOperation()->moveBefore(bodyBlock, bodyBlock->end());
    }
  }

  return mlir::success();
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createScheduleParallelizationPass() {
  return std::make_unique<ScheduleParallelizationPass>();
}

std::unique_ptr<mlir::Pass> createScheduleParallelizationPass(
    const ScheduleParallelizationPassOptions &options) {
  return std::make_unique<ScheduleParallelizationPass>(options);
}
} // namespace mlir::bmodelica
