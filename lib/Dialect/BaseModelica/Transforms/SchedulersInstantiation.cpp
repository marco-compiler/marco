#include "marco/Dialect/BaseModelica/Transforms/SchedulersInstantiation.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/Runtime/IR/Runtime.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_SCHEDULERSINSTANTIATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class SchedulersInstantiationPass
      : public impl::SchedulersInstantiationPassBase<
            SchedulersInstantiationPass>
  {
    public:
      using SchedulersInstantiationPassBase<SchedulersInstantiationPass>
          ::SchedulersInstantiationPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult processScheduleOp(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ScheduleOp scheduleOp);

      mlir::LogicalResult processInitialOp(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ScheduleOp scheduleOp,
          InitialOp initialOp);

      mlir::LogicalResult processDynamicOp(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ScheduleOp scheduleOp,
          DynamicOp dynamicOp);

      mlir::LogicalResult processParallelOps(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          llvm::ArrayRef<ParallelScheduleBlocksOp> parallelOps,
          llvm::StringRef schedulerBaseName,
          llvm::function_ref<mlir::Block*(
              mlir::OpBuilder&, mlir::Location)> createBeginFn,
          llvm::function_ref<mlir::Block*(
              mlir::OpBuilder&, mlir::Location)> createEndFn);

      void mergeVariableAccesses(
          llvm::SmallVectorImpl<mlir::Attribute>& writtenVariables,
          llvm::SmallVectorImpl<mlir::Attribute>& readVariables,
          llvm::ArrayRef<ScheduleBlockOp> blocks);

      mlir::runtime::SchedulerOp declareScheduler(
          mlir::OpBuilder& builder,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          llvm::StringRef baseName);

      mlir::LogicalResult configureScheduler(
          mlir::OpBuilder& builder,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          llvm::StringRef schedulerName,
          llvm::ArrayRef<ScheduleBlockOp> blocks,
          llvm::function_ref<mlir::Block*(
              mlir::OpBuilder&, mlir::Location)> createBeginFn);

      mlir::LogicalResult destroyScheduler(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          llvm::StringRef schedulerName,
          llvm::function_ref<mlir::Block*(
              mlir::OpBuilder&, mlir::Location)> createEndFn);

      mlir::runtime::EquationFunctionOp createWrapperFunction(
          mlir::OpBuilder& builder,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          EquationCallOp callOp);
  };
}

void SchedulersInstantiationPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::IRRewriter rewriter(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;
  llvm::SmallVector<ScheduleOp> scheduleOps;

  for (ModelOp modelOp : moduleOp.getOps<ModelOp>()) {
    for (ScheduleOp scheduleOp : modelOp.getOps<ScheduleOp>()) {
      scheduleOps.push_back(scheduleOp);
    }
  }

  for (ScheduleOp scheduleOp : scheduleOps) {
    if (mlir::failed(processScheduleOp(
            rewriter, symbolTableCollection, moduleOp, scheduleOp))) {
      return signalPassFailure();
    }
  }
}

mlir::LogicalResult SchedulersInstantiationPass::processScheduleOp(
    mlir::IRRewriter& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
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
            rewriter, symbolTableCollection, moduleOp, scheduleOp,
            initialOp))) {
      return mlir::failure();
    }
  }

  for (DynamicOp dynamicOp : dynamicOps) {
    if (mlir::failed(processDynamicOp(
            rewriter, symbolTableCollection, moduleOp, scheduleOp,
            dynamicOp))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

static void collectEquationCallBlocks(
    llvm::SmallVectorImpl<ScheduleBlockOp>& blocks,
    ParallelScheduleBlocksOp parallelOp)
{
  for (ScheduleBlockOp block : parallelOp.getOps<ScheduleBlockOp>()) {
    if (block.getBody()->getOperations().size() == 1) {
      if (mlir::isa<EquationCallOp>(block.getBody()->front())) {
        blocks.push_back(block);
      }
    }
  }
}

mlir::LogicalResult SchedulersInstantiationPass::processInitialOp(
    mlir::IRRewriter& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ScheduleOp scheduleOp,
    InitialOp initialOp)
{
  auto modelOp = scheduleOp->getParentOfType<ModelOp>();
  llvm::SmallVector<ParallelScheduleBlocksOp> parallelOps;

  for (ParallelScheduleBlocksOp parallelOp :
       initialOp.getOps<ParallelScheduleBlocksOp>()) {
    parallelOps.push_back(parallelOp);
  }

  auto createBeginFn =
      [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc) -> mlir::Block* {
    mlir::OpBuilder::InsertionGuard guard(nestedBuilder);
    nestedBuilder.setInsertionPointToEnd(moduleOp.getBody());
    auto beginFn = nestedBuilder.create<mlir::runtime::ICModelBeginOp>(loc);
    return nestedBuilder.createBlock(&beginFn.getBodyRegion());
  };

  auto createEndFn =
      [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc) -> mlir::Block* {
    mlir::OpBuilder::InsertionGuard guard(nestedBuilder);
    nestedBuilder.setInsertionPointToEnd(moduleOp.getBody());
    auto endFn = nestedBuilder.create<mlir::runtime::ICModelEndOp>(loc);
    return nestedBuilder.createBlock(&endFn.getBodyRegion());
  };

  return processParallelOps(
      rewriter, symbolTableCollection, moduleOp, parallelOps,
      modelOp.getSymName().str() + "_ic_scheduler",
      createBeginFn, createEndFn);
}

mlir::LogicalResult SchedulersInstantiationPass::processDynamicOp(
    mlir::IRRewriter& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ScheduleOp scheduleOp,
    DynamicOp dynamicOp)
{
  auto modelOp = scheduleOp->getParentOfType<ModelOp>();
  llvm::SmallVector<ParallelScheduleBlocksOp> parallelOps;

  for (ParallelScheduleBlocksOp parallelOp :
       dynamicOp.getOps<ParallelScheduleBlocksOp>()) {
    parallelOps.push_back(parallelOp);
  }

  auto createBeginFn =
      [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc) -> mlir::Block* {
        mlir::OpBuilder::InsertionGuard guard(nestedBuilder);
        nestedBuilder.setInsertionPointToEnd(moduleOp.getBody());

        auto beginFn =
            nestedBuilder.create<mlir::runtime::DynamicModelBeginOp>(loc);

        return nestedBuilder.createBlock(&beginFn.getBodyRegion());
      };

  auto createEndFn =
      [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc) -> mlir::Block* {
        mlir::OpBuilder::InsertionGuard guard(nestedBuilder);
        nestedBuilder.setInsertionPointToEnd(moduleOp.getBody());

        auto endFn =
            nestedBuilder.create<mlir::runtime::DynamicModelEndOp>(loc);

        return nestedBuilder.createBlock(&endFn.getBodyRegion());
      };

  return processParallelOps(
      rewriter, symbolTableCollection, moduleOp, parallelOps,
      modelOp.getSymName().str() + "_dynamic_scheduler",
      createBeginFn, createEndFn);
}

static bool hasSingleScalarEquationCall(ScheduleBlockOp blockOp)
{
  llvm::SmallVector<EquationCallOp, 1> callOps;

  for (EquationCallOp callOp : blockOp.getOps<EquationCallOp>()) {
    callOps.push_back(callOp);
  }

  if (callOps.size() != 1) {
    return false;
  }

  if (auto indices = callOps[0].getIndices()) {
    return indices->getValue().flatSize() == 1;
  }

  return true;
}

mlir::LogicalResult SchedulersInstantiationPass::processParallelOps(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    llvm::ArrayRef<ParallelScheduleBlocksOp> parallelOps,
    llvm::StringRef schedulerBaseName,
    llvm::function_ref<mlir::Block*(
        mlir::OpBuilder&, mlir::Location)> createBeginFn,
    llvm::function_ref<mlir::Block*(
        mlir::OpBuilder&, mlir::Location)> createEndFn)
{
  for (ParallelScheduleBlocksOp parallelOp : parallelOps) {
    llvm::SmallVector<ScheduleBlockOp, 10> blocks;
    collectEquationCallBlocks(blocks, parallelOp);

    if (blocks.size() == 1 && hasSingleScalarEquationCall(blocks[0])) {
      continue;
    }

    llvm::SmallVector<mlir::Attribute> writtenVariables;
    llvm::SmallVector<mlir::Attribute> readVariables;
    mergeVariableAccesses(writtenVariables, readVariables, blocks);

    rewriter.setInsertionPointToEnd(parallelOp.getBody());

    auto newScheduleOp = rewriter.create<ScheduleBlockOp>(
        parallelOp.getLoc(), false,
        rewriter.getArrayAttr(writtenVariables),
        rewriter.getArrayAttr(readVariables));

    rewriter.createBlock(&newScheduleOp.getBodyRegion());
    rewriter.setInsertionPointToStart(newScheduleOp.getBody());

    mlir::runtime::SchedulerOp schedulerOp = declareScheduler(
        rewriter, symbolTableCollection, moduleOp, parallelOp.getLoc(),
        schedulerBaseName);

    if (!schedulerOp) {
      return mlir::failure();
    }

    if (mlir::failed(configureScheduler(
            rewriter, symbolTableCollection, moduleOp, parallelOp.getLoc(),
            schedulerOp.getSymName(), blocks, createBeginFn))) {
      return mlir::failure();
    }

    if (mlir::failed(destroyScheduler(
            rewriter, parallelOp.getLoc(), schedulerOp.getSymName(),
            createEndFn))) {
      return mlir::failure();
    }

    rewriter.create<mlir::runtime::SchedulerRunOp>(
        parallelOp.getLoc(), schedulerOp.getSymName());

    for (ScheduleBlockOp block : blocks) {
      rewriter.eraseOp(block);
    }
  }

  return mlir::success();
}

void SchedulersInstantiationPass::mergeVariableAccesses(
    llvm::SmallVectorImpl<mlir::Attribute>& writtenVariables,
    llvm::SmallVectorImpl<mlir::Attribute>& readVariables,
    llvm::ArrayRef<ScheduleBlockOp> blocks)
{
  llvm::DenseMap<mlir::SymbolRefAttr, IndexSet> writtenVariablesIndices;
  llvm::DenseMap<mlir::SymbolRefAttr, IndexSet> readVariablesIndices;

  for (ScheduleBlockOp block : blocks) {
    for (VariableAttr variable :
         block.getWrittenVariables().getAsRange<VariableAttr>()) {
      writtenVariablesIndices[variable.getName()] +=
          variable.getIndices().getValue();
    }

    for (VariableAttr variable :
         block.getReadVariables().getAsRange<VariableAttr>()) {
      readVariablesIndices[variable.getName()] +=
          variable.getIndices().getValue();
    }
  }

  for (const auto& entry : writtenVariablesIndices) {
    writtenVariables.push_back(VariableAttr::get(
        &getContext(), entry.getFirst(),
        IndexSetAttr::get(&getContext(), entry.getSecond())));
  }

  for (const auto& entry : readVariablesIndices) {
    readVariables.push_back(VariableAttr::get(
        &getContext(), entry.getFirst(),
        IndexSetAttr::get(&getContext(), entry.getSecond())));
  }

  auto sortFn = [](mlir::Attribute first, mlir::Attribute second) -> bool {
    auto firstName = first.cast<VariableAttr>().getName();
    auto secondName = second.cast<VariableAttr>().getName();

    auto firstRoot = firstName.getRootReference().getValue();
    auto secondRoot = secondName.getRootReference().getValue();

    if (firstRoot != secondRoot) {
      return firstRoot < secondRoot;
    }

    size_t firstLength = firstName.getNestedReferences().size();
    size_t secondLength = secondName.getNestedReferences().size();

    for (size_t i = 0, e = std::min(firstLength, secondLength); i < e; ++i) {
      auto firstNested = firstName.getNestedReferences()[i].getValue();
      auto secondNested = secondName.getNestedReferences()[i].getValue();

      if (firstNested != secondNested) {
        return firstNested < secondNested;
      }
    }

    return firstLength <= secondLength;
  };

  std::sort(writtenVariables.begin(), writtenVariables.end(), sortFn);
  std::sort(readVariables.begin(), readVariables.end(), sortFn);
}

mlir::runtime::SchedulerOp SchedulersInstantiationPass::declareScheduler(
    mlir::OpBuilder& builder,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    llvm::StringRef baseName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto schedulerOp =
      builder.create<mlir::runtime::SchedulerOp>(loc, baseName);

  symbolTableCollection.getSymbolTable(moduleOp).insert(schedulerOp);
  return schedulerOp;
}

mlir::LogicalResult SchedulersInstantiationPass::configureScheduler(
    mlir::OpBuilder& builder,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    llvm::StringRef schedulerName,
    llvm::ArrayRef<ScheduleBlockOp> blocks,
    llvm::function_ref<mlir::Block*(
        mlir::OpBuilder&, mlir::Location)> createBeginFn)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  llvm::SmallVector<EquationCallOp> equationCallOps;

  for (ScheduleBlockOp block : blocks) {
    block.walk([&](EquationCallOp callOp) {
      equationCallOps.push_back(callOp);
    });
  }

  mlir::Block* beginFnBody = createBeginFn(builder, loc);
  builder.setInsertionPointToStart(beginFnBody);

  builder.create<mlir::runtime::SchedulerCreateOp>(loc, schedulerName);

  for (EquationCallOp callOp : equationCallOps) {
    MultidimensionalRangeAttr ranges = nullptr;

    if (callOp.getIndices()) {
      ranges = callOp.getIndicesAttr();
    }

    mlir::runtime::EquationFunctionOp wrapperFunction =
        createWrapperFunction(builder, symbolTableCollection, moduleOp, callOp);

    if (!wrapperFunction) {
      return mlir::failure();
    }

    builder.create<mlir::runtime::SchedulerAddEquationOp>(
        callOp.getLoc(), schedulerName, wrapperFunction.getSymName(), ranges,
        callOp.getParallelizable());
  }

  return mlir::success();
}

mlir::LogicalResult SchedulersInstantiationPass::destroyScheduler(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    llvm::StringRef schedulerName,
    llvm::function_ref<mlir::Block*(
        mlir::OpBuilder&, mlir::Location)> createEndFn)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Block* endFnBody = createEndFn(builder, loc);
  builder.setInsertionPointToStart(endFnBody);
  builder.create<mlir::runtime::SchedulerDestroyOp>(loc, schedulerName);
  return mlir::success();
}

mlir::runtime::EquationFunctionOp
SchedulersInstantiationPass::createWrapperFunction(
    mlir::OpBuilder& builder,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    EquationCallOp callOp)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  uint64_t numOfInductions = 0;

  if (auto indicesAttr = callOp.getIndices()) {
    numOfInductions = indicesAttr->getValue().rank();
  }

  std::string functionName = callOp.getCallee().str() + "_wrapper";

  auto wrapperFunction = builder.create<mlir::runtime::EquationFunctionOp>(
      callOp.getLoc(), functionName, numOfInductions);

  symbolTableCollection.getSymbolTable(moduleOp).insert(wrapperFunction);
  mlir::Block* entryBlock = wrapperFunction.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto equationFunctionOp =
      symbolTableCollection.lookupSymbolIn<EquationFunctionOp>(
          moduleOp, callOp.getCalleeAttr());

  builder.create<CallOp>(
      callOp.getLoc(), equationFunctionOp, wrapperFunction.getArguments());

  builder.create<mlir::runtime::ReturnOp>(callOp.getLoc(), std::nullopt);
  return wrapperFunction;
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createSchedulersInstantiationPass()
  {
    return std::make_unique<SchedulersInstantiationPass>();
  }
}
