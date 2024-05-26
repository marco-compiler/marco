#include "marco/Dialect/BaseModelica/Transforms/ScheduleParallelization.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelicaDialect.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_SCHEDULEPARALLELIZATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class ScheduleParallelizationPass
      : public mlir::bmodelica::impl::ScheduleParallelizationPassBase<
            ScheduleParallelizationPass>
  {
    public:
      static const int64_t kUnlimitedGroupBlocks = -1;

    public:
      using ScheduleParallelizationPassBase<ScheduleParallelizationPass>
          ::ScheduleParallelizationPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult processScheduleOp(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          ScheduleOp scheduleOp);

      mlir::LogicalResult processInitialOp(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          InitialOp initialOp);

      mlir::LogicalResult processDynamicOp(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          DynamicOp dynamicOp);

      mlir::LogicalResult parallelizeBlocks(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          llvm::ArrayRef<ScheduleBlockOp> blocks);
  };
}

void ScheduleParallelizationPass::runOnOperation()
{
  ModelOp modelOp = getOperation();
  mlir::SymbolTableCollection symbolTableCollection;

  for (ScheduleOp scheduleOp : modelOp.getOps<ScheduleOp>()) {
    if (mlir::failed(processScheduleOp(
            symbolTableCollection, modelOp, scheduleOp))) {
      return signalPassFailure();
    }
  }
}

mlir::LogicalResult ScheduleParallelizationPass::processScheduleOp(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    ScheduleOp scheduleOp)
{
  llvm::SmallVector<InitialOp> initialOps;
  llvm::SmallVector<DynamicOp> dynamicOps;

  for (auto& op : scheduleOp.getOps()) {
    if (auto initialOp = mlir::dyn_cast<InitialOp>(op)) {
      initialOps.push_back(initialOp);
      continue;
    }

    if (auto dynamicOp = mlir::dyn_cast<DynamicOp>(op)) {
      dynamicOps.push_back(dynamicOp);
      continue;
    }
  }

  for (InitialOp initialOp : initialOps) {
    if (mlir::failed(processInitialOp(
            symbolTableCollection, modelOp, initialOp))) {
      return mlir::failure();
    }
  }

  for (DynamicOp dynamicOp : dynamicOps) {
    if (mlir::failed(processDynamicOp(
            symbolTableCollection, modelOp, dynamicOp))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult ScheduleParallelizationPass::processInitialOp(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    InitialOp initialOp)
{
  llvm::SmallVector<ScheduleBlockOp> blocks;

  for (auto& op : llvm::make_early_inc_range(initialOp.getOps())) {
    if (auto blockOp = mlir::dyn_cast<ScheduleBlockOp>(op)) {
      if (blockOp.getParallelizable()) {
        blocks.push_back(blockOp);
      } else {
        if (mlir::failed(parallelizeBlocks(
                symbolTableCollection, modelOp, blocks))) {
          return mlir::failure();
        }

        blocks.clear();
      }
    } else {
      if (mlir::failed(parallelizeBlocks(
              symbolTableCollection, modelOp, blocks))) {
        return mlir::failure();
      }

      blocks.clear();
    }
  }

  // Parallelize the last chunk of blocks.
  return parallelizeBlocks(symbolTableCollection, modelOp, blocks);
}

mlir::LogicalResult ScheduleParallelizationPass::processDynamicOp(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    DynamicOp dynamicOp)
{
  llvm::SmallVector<ScheduleBlockOp> blocks;

  for (auto& op : llvm::make_early_inc_range(dynamicOp.getOps())) {
    if (auto blockOp = mlir::dyn_cast<ScheduleBlockOp>(op)) {
      if (blockOp.getParallelizable()) {
        blocks.push_back(blockOp);
      } else {
        if (mlir::failed(parallelizeBlocks(
                symbolTableCollection, modelOp, blocks))) {
          return mlir::failure();
        }

        blocks.clear();
      }
    } else {
      if (mlir::failed(parallelizeBlocks(
              symbolTableCollection, modelOp, blocks))) {
        return mlir::failure();
      }

      blocks.clear();
    }
  }

  // Parallelize the last chunk of blocks.
  return parallelizeBlocks(symbolTableCollection, modelOp, blocks);
}

mlir::LogicalResult ScheduleParallelizationPass::parallelizeBlocks(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    llvm::ArrayRef<ScheduleBlockOp> blocks)
{
  if (blocks.empty()) {
    return mlir::success();
  }

  // Compute the writes map.
  WritesMap<VariableOp, ScheduleBlockOp> writesMap;

  if (mlir::failed(getWritesMap(
          writesMap, modelOp, blocks, symbolTableCollection))) {
    return mlir::failure();
  }

  // Compute the outgoing arcs and the in-degree of each block.
  llvm::DenseMap<
      ScheduleBlockOp,
      llvm::DenseSet<ScheduleBlockOp>> dependantBlocks;

  llvm::DenseMap<ScheduleBlockOp, size_t> inDegrees;

  for (ScheduleBlockOp block : blocks) {
    inDegrees[block] = 0;
  }

  for (ScheduleBlockOp readingBlock : blocks) {
    for (auto readVariable :
         readingBlock.getReadVariables().getAsRange<VariableAttr>()) {
      auto readVariableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
          modelOp, readVariable.getName());

      for (const auto& writingBlock :
           llvm::make_range(writesMap.equal_range(readVariableOp))) {
        if (writingBlock.second.second == readingBlock) {
          // Ignore self-loops.
          continue;
        }

        if (writingBlock.second.first.overlaps(
                readVariable.getIndices().getValue()) ||
            (writingBlock.second.first.empty() &&
             readVariable.getIndices().getValue().empty())) {
          dependantBlocks[writingBlock.second.second].insert(readingBlock);
          inDegrees[readingBlock]++;
        }
      }
    }
  }

  // Compute the sets of independent blocks.
  llvm::SmallVector<llvm::SmallVector<ScheduleBlockOp, 10>, 10> groups;
  llvm::DenseSet<ScheduleBlockOp> currentBlocks;
  llvm::DenseSet<ScheduleBlockOp> newBlocks;

  for (ScheduleBlockOp block : blocks) {
    if (inDegrees[block] == 0) {
      currentBlocks.insert(block);
    }
  }

  while (!currentBlocks.empty()) {
    llvm::SmallVector<ScheduleBlockOp, 10> independentBlocks;

    for (ScheduleBlockOp block : currentBlocks) {
      if (inDegrees[block] == 0 &&
          (maxParallelBlocks == kUnlimitedGroupBlocks ||
           static_cast<int64_t>(independentBlocks.size()) <
               maxParallelBlocks)) {
        independentBlocks.push_back(block);

        for (ScheduleBlockOp dependantBlock : dependantBlocks[block]) {
          assert(inDegrees[dependantBlock] > 0);
          inDegrees[dependantBlock]--;
          newBlocks.insert(dependantBlock);

          // Avoid visiting again the block at the next iteration.
          newBlocks.erase(block);
        }
      } else {
        newBlocks.insert(block);
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

  for (const auto& group : groups) {
    auto parallelBlocksOp =
        rewriter.create<ParallelScheduleBlocksOp>(modelOp.getLoc());

    mlir::OpBuilder::InsertionGuard guard(rewriter);

    mlir::Block* bodyBlock =
        rewriter.createBlock(&parallelBlocksOp.getBodyRegion());

    for (ScheduleBlockOp block : group) {
      block.getOperation()->moveBefore(bodyBlock, bodyBlock->end());
    }
  }

  return mlir::success();
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createScheduleParallelizationPass()
  {
    return std::make_unique<ScheduleParallelizationPass>();
  }

  std::unique_ptr<mlir::Pass> createScheduleParallelizationPass(
      const ScheduleParallelizationPassOptions& options)
  {
    return std::make_unique<ScheduleParallelizationPass>(options);
  }
}
