#include "marco/Codegen/Conversion/BaseModelicaToCF/BaseModelicaToCF.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/IR/VariablesDimensionsDependencyGraph.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
#define GEN_PASS_DEF_BASEMODELICATOCFCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
} // namespace mlir

using namespace ::mlir::bmodelica;

namespace {
struct ConversionInstance {
  FunctionOp functionOp;
  llvm::SmallVector<VariableOp> inputVariables;
  llvm::SmallVector<VariableOp> outputVariables;
  llvm::SmallVector<VariableOp> protectedVariables;
  RawFunctionOp rawFunctionOp;
};
} // namespace

namespace {
class BaseModelicaToCFConversionPass
    : public mlir::impl::BaseModelicaToCFConversionPassBase<
          BaseModelicaToCFConversionPass> {
public:
  using BaseModelicaToCFConversionPassBase<
      BaseModelicaToCFConversionPass>::BaseModelicaToCFConversionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult
  moveAndConvertFunctionBody(const ConversionInstance &conversionInstance);

  RawVariableOp createVariable(mlir::RewriterBase &rewriter,
                               VariableOp variableOp, mlir::Block *exitBlock,
                               mlir::Block *&lastBlockBeforeExitBlock);

  /// Replace the references to the symbol of a variable with references to
  /// the SSA value of its equivalent "raw" variable.
  void replaceSymbolAccesses(
      mlir::RewriterBase &rewriter, mlir::Region &region,
      const llvm::StringMap<mlir::BlockArgument> &inputVars,
      const llvm::StringMap<RawVariableOp> &outputAndProtectedVars);

  void replaceSymbolAccess(
      mlir::RewriterBase &rewriter, mlir::Operation *op,
      const llvm::StringMap<mlir::BlockArgument> &inputVars,
      const llvm::StringMap<RawVariableOp> &outputAndProtectedVars);

  void replaceSymbolAccess(
      mlir::RewriterBase &rewriter, VariableGetOp op,
      const llvm::StringMap<mlir::BlockArgument> &inputVars,
      const llvm::StringMap<RawVariableOp> &outputAndProtectedVars);

  void replaceSymbolAccess(
      mlir::RewriterBase &rewriter, VariableSetOp op,
      const llvm::StringMap<RawVariableOp> &outputAndProtectedVars);

  mlir::LogicalResult createCFG(mlir::RewriterBase &rewriter,
                                mlir::Operation *op, mlir::Block *loopExitBlock,
                                mlir::Block *functionReturnBlock);

  mlir::LogicalResult createCFG(mlir::RewriterBase &rewriter, BreakOp op,
                                mlir::Block *loopExitBlock);

  mlir::LogicalResult createCFG(mlir::RewriterBase &rewriter, ForOp op,
                                mlir::Block *functionReturnBlock);

  mlir::LogicalResult createCFG(mlir::RewriterBase &rewriter, IfOp op,
                                mlir::Block *loopExitBlock,
                                mlir::Block *functionReturnBlock);

  mlir::LogicalResult createCFG(mlir::RewriterBase &rewriter, WhileOp op,
                                mlir::Block *functionReturnBlock);

  mlir::LogicalResult createCFG(mlir::RewriterBase &rewriter, ReturnOp op,
                                mlir::Block *functionReturnBlock);

  mlir::LogicalResult recurse(mlir::RewriterBase &rewriter, mlir::Block *first,
                              mlir::Block *last, mlir::Block *loopExitBlock,
                              mlir::Block *functionReturnBlock);
};
} // namespace

namespace {
void collectVariables(FunctionOp functionOp,
                      llvm::SmallVectorImpl<VariableOp> &inputVariables,
                      llvm::SmallVectorImpl<VariableOp> &outputVariables,
                      llvm::SmallVectorImpl<VariableOp> &protectedVariables) {
  for (VariableOp variableOp : functionOp.getVariables()) {
    if (variableOp.isInput()) {
      inputVariables.push_back(variableOp);
    } else if (variableOp.isOutput()) {
      outputVariables.push_back(variableOp);
    } else {
      protectedVariables.push_back(variableOp);
    }
  }
}

RawFunctionOp declareRawFunctionOp(FunctionOp functionOp,
                                   llvm::ArrayRef<VariableOp> inputVariables,
                                   llvm::ArrayRef<VariableOp> outputVariables) {
  mlir::OpBuilder builder(functionOp);

  // Determine the signature of the function.
  llvm::SmallVector<mlir::Type> argTypes;
  llvm::SmallVector<mlir::Type> resultTypes;

  for (VariableOp variableOp : inputVariables) {
    mlir::Type unwrappedType = variableOp.getVariableType().unwrap();
    argTypes.push_back(unwrappedType);
  }

  for (VariableOp variableOp : outputVariables) {
    mlir::Type unwrappedType = variableOp.getVariableType().unwrap();
    resultTypes.push_back(unwrappedType);
  }

  // Declare the function.
  return builder.create<RawFunctionOp>(
      functionOp.getLoc(), functionOp.getSymName(),
      builder.getFunctionType(argTypes, resultTypes));
}
} // namespace

void BaseModelicaToCFConversionPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  llvm::SmallVector<ConversionInstance> conversions;

  // Declare the raw functions.
  for (FunctionOp functionOp :
       llvm::make_early_inc_range(moduleOp.getOps<FunctionOp>())) {
    auto &conversionInstance = conversions.emplace_back();
    conversionInstance.functionOp = functionOp;

    collectVariables(functionOp, conversionInstance.inputVariables,
                     conversionInstance.outputVariables,
                     conversionInstance.protectedVariables);

    conversionInstance.rawFunctionOp =
        declareRawFunctionOp(functionOp, conversionInstance.inputVariables,
                             conversionInstance.outputVariables);
  }

  // Populate the body of raw functions.
  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), conversions,
          [&](const ConversionInstance &conversion) {
            return moveAndConvertFunctionBody(conversion);
          }))) {
    return signalPassFailure();
  }

  // Erase the original functions.
  for (const ConversionInstance &conversion : conversions) {
    conversion.functionOp->erase();
  }
}

mlir::LogicalResult BaseModelicaToCFConversionPass::moveAndConvertFunctionBody(
    const ConversionInstance &conversionInstance) {
  FunctionOp functionOp = conversionInstance.functionOp;
  RawFunctionOp rawFunctionOp = conversionInstance.rawFunctionOp;

  mlir::IRRewriter rewriter(rawFunctionOp);
  mlir::Location loc = functionOp.getLoc();
  mlir::SymbolTable symbolTable(functionOp.getOperation());

  // Add the entry block and map the arguments to the input variables.
  mlir::Block *entryBlock = rawFunctionOp.addEntryBlock();
  mlir::Block *lastBlockBeforeExitBlock = entryBlock;

  llvm::StringMap<mlir::BlockArgument> argsMapping;

  for (const auto &[var, arg] : llvm::zip(conversionInstance.inputVariables,
                                          rawFunctionOp.getArguments())) {
    VariableOp variableOp = var;
    argsMapping[variableOp.getSymName()] = arg;
  }

  // Create the return block.
  mlir::Block *exitBlock = rewriter.createBlock(
      &rawFunctionOp.getFunctionBody(), rawFunctionOp.getFunctionBody().end());

  // Create the variables. The order in which the variables are created has
  // to take into account the dependencies of their dynamic dimensions. At
  // the same time, however, the relative order of the output variables
  // must be the same as the declared one, in order to preserve the
  // correctness of the calls. This last aspect is already taken into
  // account by the dependency graph.
  VariablesDimensionsDependencyGraph dimensionsGraph;

  dimensionsGraph.addVariables(conversionInstance.outputVariables);
  dimensionsGraph.addVariables(conversionInstance.protectedVariables);

  dimensionsGraph.discoverDependencies();

  llvm::StringMap<RawVariableOp> rawVariables;

  for (VariableOp variable : dimensionsGraph.postOrder()) {
    rawVariables[variable.getSymName()] =
        createVariable(rewriter, variable, exitBlock, lastBlockBeforeExitBlock);
  }

  // Convert the algorithms.
  for (AlgorithmOp algorithmOp : functionOp.getOps<AlgorithmOp>()) {
    mlir::Region &algorithmRegion = algorithmOp.getBodyRegion();

    if (algorithmRegion.empty()) {
      continue;
    }

    rewriter.setInsertionPointToEnd(lastBlockBeforeExitBlock);
    rewriter.create<mlir::cf::BranchOp>(loc, &algorithmRegion.front());

    if (mlir::failed(recurse(rewriter, &algorithmRegion.front(),
                             &algorithmRegion.back(), nullptr, exitBlock))) {
      return mlir::failure();
    }

    lastBlockBeforeExitBlock = &algorithmRegion.back();
    rewriter.inlineRegionBefore(algorithmRegion, exitBlock);
  }

  rewriter.setInsertionPointToEnd(lastBlockBeforeExitBlock);
  rewriter.create<mlir::cf::BranchOp>(loc, exitBlock);

  // Replace symbol uses with SSA uses.
  replaceSymbolAccesses(rewriter, rawFunctionOp.getFunctionBody(), argsMapping,
                        rawVariables);

  // Populate the exit block.
  rewriter.setInsertionPointToStart(exitBlock);
  llvm::SmallVector<mlir::Value> results;

  for (VariableOp variableOp : conversionInstance.outputVariables) {
    RawVariableOp rawVariableOp = rawVariables[variableOp.getSymName()];

    results.push_back(
        rewriter.create<RawVariableGetOp>(variableOp->getLoc(), rawVariableOp));
  }

  rewriter.create<RawReturnOp>(loc, results);

  // Erase the unreachable blocks.
  (void)mlir::eraseUnreachableBlocks(rewriter, rawFunctionOp->getRegions());

  return mlir::success();
}

RawVariableOp BaseModelicaToCFConversionPass::createVariable(
    mlir::RewriterBase &rewriter, VariableOp variableOp, mlir::Block *exitBlock,
    mlir::Block *&lastBlockBeforeExitBlock) {
  // Inline the operations to compute the dimensions constraints, if any.
  mlir::Region &constraintsRegion = variableOp.getConstraintsRegion();

  // The YieldOp of the constraints region will be erased, so we need to
  // store the list of its operands elsewhere.
  llvm::SmallVector<mlir::Value> constraints;

  if (!constraintsRegion.empty()) {
    auto constraintsTerminator =
        mlir::cast<YieldOp>(constraintsRegion.back().getTerminator());

    for (mlir::Value constraint : constraintsTerminator.getValues()) {
      constraints.push_back(constraint);
    }

    rewriter.eraseOp(constraintsTerminator);

    rewriter.setInsertionPointToEnd(lastBlockBeforeExitBlock);

    rewriter.create<mlir::cf::BranchOp>(constraintsRegion.getLoc(),
                                        &constraintsRegion.back());

    lastBlockBeforeExitBlock = &constraintsRegion.back();
    rewriter.inlineRegionBefore(constraintsRegion, exitBlock);
  }

  // Create the block containing the declaration of the variable.
  mlir::Block *variableBlock = rewriter.createBlock(exitBlock);
  rewriter.setInsertionPointToEnd(lastBlockBeforeExitBlock);
  rewriter.create<mlir::cf::BranchOp>(variableOp.getLoc(), variableBlock);
  lastBlockBeforeExitBlock = variableBlock;

  rewriter.setInsertionPointToStart(variableBlock);

  return rewriter.create<RawVariableOp>(
      variableOp.getLoc(), variableOp.getVariableType().toTensorType(),
      variableOp.getSymName(), variableOp.getDimensionsConstraints(),
      constraints, variableOp.isOutput());
}

void BaseModelicaToCFConversionPass::replaceSymbolAccesses(
    mlir::RewriterBase &rewriter, mlir::Region &region,
    const llvm::StringMap<mlir::BlockArgument> &inputVars,
    const llvm::StringMap<RawVariableOp> &outputAndProtectedVars) {
  region.walk([&](mlir::Operation *op) {
    return replaceSymbolAccess(rewriter, op, inputVars, outputAndProtectedVars);
  });
}

void BaseModelicaToCFConversionPass::replaceSymbolAccess(
    mlir::RewriterBase &rewriter, mlir::Operation *op,
    const llvm::StringMap<mlir::BlockArgument> &inputVars,
    const llvm::StringMap<RawVariableOp> &outputAndProtectedVars) {
  if (auto castedOp = mlir::dyn_cast<VariableGetOp>(op)) {
    return replaceSymbolAccess(rewriter, castedOp, inputVars,
                               outputAndProtectedVars);
  }

  if (auto castedOp = mlir::dyn_cast<VariableSetOp>(op)) {
    return replaceSymbolAccess(rewriter, castedOp, outputAndProtectedVars);
  }
}

void BaseModelicaToCFConversionPass::replaceSymbolAccess(
    mlir::RewriterBase &rewriter, VariableGetOp op,
    const llvm::StringMap<mlir::BlockArgument> &inputVars,
    const llvm::StringMap<RawVariableOp> &outputAndProtectedVars) {
  rewriter.setInsertionPoint(op);
  auto inputVarIt = inputVars.find(op.getVariable());

  if (inputVarIt != inputVars.end()) {
    rewriter.replaceOp(op, inputVarIt->getValue());
  } else {
    auto writableVarIt = outputAndProtectedVars.find(op.getVariable());
    assert(writableVarIt != outputAndProtectedVars.end());
    RawVariableOp rawVariableOp = writableVarIt->getValue();
    rewriter.replaceOpWithNewOp<RawVariableGetOp>(op, rawVariableOp);
  }
}

void BaseModelicaToCFConversionPass::replaceSymbolAccess(
    mlir::RewriterBase &rewriter, VariableSetOp op,
    const llvm::StringMap<RawVariableOp> &outputAndProtectedVars) {
  rewriter.setInsertionPoint(op);
  auto it = outputAndProtectedVars.find(op.getVariable());
  assert(it != outputAndProtectedVars.end());
  RawVariableOp rawVariableOp = it->getValue();

  mlir::Value value = op.getValue();

  if (auto indices = op.getIndices(); !indices.empty()) {
    mlir::Value tensor =
        rewriter.create<RawVariableGetOp>(op.getLoc(), rawVariableOp);

    mlir::Type tensorElementType =
        mlir::cast<mlir::TensorType>(tensor.getType()).getElementType();

    if (value.getType() != tensorElementType) {
      value = rewriter.create<CastOp>(op.getLoc(), tensorElementType, value);
    }

    tensor =
        rewriter.create<TensorInsertOp>(op.getLoc(), value, tensor, indices);

    rewriter.replaceOpWithNewOp<RawVariableSetOp>(op, rawVariableOp, tensor);
  } else {
    auto rawVariableTensorType =
        mlir::cast<mlir::TensorType>(rawVariableOp.getVariable().getType());

    if (rawVariableTensorType.getShape().empty()) {
      mlir::Type elementType = rawVariableTensorType.getElementType();

      if (value.getType() != elementType) {
        value = rewriter.create<CastOp>(op.getLoc(), elementType, value);
      }
    } else {
      if (value.getType() != rawVariableTensorType) {
        value =
            rewriter.create<CastOp>(op.getLoc(), rawVariableTensorType, value);
      }
    }

    rewriter.replaceOpWithNewOp<RawVariableSetOp>(op, rawVariableOp, value);
  }
}

mlir::LogicalResult BaseModelicaToCFConversionPass::createCFG(
    mlir::RewriterBase &rewriter, mlir::Operation *op,
    mlir::Block *loopExitBlock, mlir::Block *functionReturnBlock) {
  if (auto breakOp = mlir::dyn_cast<BreakOp>(op)) {
    return createCFG(rewriter, breakOp, loopExitBlock);
  }

  if (auto forOp = mlir::dyn_cast<ForOp>(op)) {
    return createCFG(rewriter, forOp, functionReturnBlock);
  }

  if (auto ifOp = mlir::dyn_cast<IfOp>(op)) {
    return createCFG(rewriter, ifOp, loopExitBlock, functionReturnBlock);
  }

  if (auto whileOp = mlir::dyn_cast<WhileOp>(op)) {
    return createCFG(rewriter, whileOp, functionReturnBlock);
  }

  if (auto returnOp = mlir::dyn_cast<ReturnOp>(op)) {
    return createCFG(rewriter, returnOp, functionReturnBlock);
  }

  return mlir::success();
}

mlir::LogicalResult BaseModelicaToCFConversionPass::createCFG(
    mlir::RewriterBase &rewriter, BreakOp op, mlir::Block *loopExitBlock) {
  if (!loopExitBlock) {
    return mlir::failure();
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  mlir::Block *currentBlock = rewriter.getInsertionBlock();
  rewriter.splitBlock(currentBlock, op->getIterator());

  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<mlir::cf::BranchOp>(op->getLoc(), loopExitBlock);

  rewriter.eraseOp(op);
  return mlir::success();
}

mlir::LogicalResult BaseModelicaToCFConversionPass::createCFG(
    mlir::RewriterBase &rewriter, ForOp op, mlir::Block *functionReturnBlock) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  // Split the current block.
  mlir::Block *currentBlock = rewriter.getInsertionBlock();

  mlir::Block *continuation =
      rewriter.splitBlock(currentBlock, op->getIterator());

  // Keep the references to the op blocks.
  mlir::Block *conditionFirst = &op.getConditionRegion().front();
  mlir::Block *conditionLast = &op.getConditionRegion().back();
  mlir::Block *bodyFirst = &op.getBodyRegion().front();
  mlir::Block *bodyLast = &op.getBodyRegion().back();
  mlir::Block *stepFirst = &op.getStepRegion().front();
  mlir::Block *stepLast = &op.getStepRegion().back();

  // Inline the regions.
  rewriter.inlineRegionBefore(op.getConditionRegion(), continuation);
  rewriter.inlineRegionBefore(op.getBodyRegion(), continuation);
  rewriter.inlineRegionBefore(op.getStepRegion(), continuation);

  // Start the for loop by branching to the "condition" region.
  rewriter.setInsertionPointToEnd(currentBlock);

  rewriter.create<mlir::cf::BranchOp>(op->getLoc(), conditionFirst,
                                      op.getArgs());

  // Check the condition.
  auto conditionOp = mlir::cast<ConditionOp>(conditionLast->getTerminator());

  rewriter.setInsertionPoint(conditionOp);

  mlir::Value conditionValue = conditionOp.getCondition();

  if (conditionValue.getType() != rewriter.getI1Type()) {
    conditionValue = rewriter.create<CastOp>(
        conditionValue.getLoc(), rewriter.getI1Type(), conditionValue);
  }

  rewriter.create<mlir::cf::CondBranchOp>(conditionOp->getLoc(), conditionValue,
                                          bodyFirst, conditionOp.getValues(),
                                          continuation, std::nullopt);

  rewriter.eraseOp(conditionOp);

  // If present, replace "body" block terminator with a branch to the
  // "step" block. If not present, just place the branch.
  rewriter.setInsertionPointToEnd(bodyLast);
  llvm::SmallVector<mlir::Value, 3> bodyYieldValues;

  if (auto yieldOp = mlir::dyn_cast<YieldOp>(bodyLast->back())) {
    for (mlir::Value value : yieldOp.getValues()) {
      bodyYieldValues.push_back(value);
    }

    rewriter.eraseOp(yieldOp);
  }

  rewriter.create<mlir::cf::BranchOp>(op->getLoc(), stepFirst, bodyYieldValues);

  // Branch to the condition check after incrementing the induction
  // variable.
  rewriter.setInsertionPointToEnd(stepLast);
  llvm::SmallVector<mlir::Value, 3> stepYieldValues;

  if (auto yieldOp = mlir::dyn_cast<YieldOp>(stepLast->back())) {
    for (mlir::Value value : yieldOp.getValues()) {
      stepYieldValues.push_back(value);
    }

    rewriter.eraseOp(yieldOp);
  }

  rewriter.create<mlir::cf::BranchOp>(op->getLoc(), conditionFirst,
                                      stepYieldValues);

  // Erase the operation.
  rewriter.eraseOp(op);

  // Recurse on the body operations.
  return recurse(rewriter, bodyFirst, bodyLast, continuation,
                 functionReturnBlock);
}

mlir::LogicalResult
BaseModelicaToCFConversionPass::createCFG(mlir::RewriterBase &rewriter, IfOp op,
                                          mlir::Block *loopExitBlock,
                                          mlir::Block *functionReturnBlock) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  // Split the current block.
  mlir::Block *currentBlock = rewriter.getInsertionBlock();

  mlir::Block *continuation =
      rewriter.splitBlock(currentBlock, op->getIterator());

  // Keep the references to the op blocks.
  mlir::Block *thenFirst = &op.getThenRegion().front();
  mlir::Block *thenLast = &op.getThenRegion().back();

  // Inline the regions.
  rewriter.inlineRegionBefore(op.getThenRegion(), continuation);
  rewriter.setInsertionPointToEnd(currentBlock);

  mlir::Value conditionValue = op.getCondition();

  if (conditionValue.getType() != rewriter.getI1Type()) {
    conditionValue = rewriter.create<CastOp>(
        conditionValue.getLoc(), rewriter.getI1Type(), conditionValue);
  }

  if (op.getElseRegion().empty()) {
    // Branch to the "then" region or to the continuation block according
    // to the condition.

    rewriter.create<mlir::cf::CondBranchOp>(op->getLoc(), conditionValue,
                                            thenFirst, std::nullopt,
                                            continuation, std::nullopt);

    rewriter.setInsertionPointToEnd(thenLast);
    rewriter.create<mlir::cf::BranchOp>(op->getLoc(), continuation);

    // Erase the operation.
    rewriter.eraseOp(op);

    // Recurse on the body operations.
    if (mlir::failed(recurse(rewriter, thenFirst, thenLast, loopExitBlock,
                             functionReturnBlock))) {
      return mlir::failure();
    }
  } else {
    // Branch to the "then" region or to the "else" region according
    // to the condition.
    mlir::Block *elseFirst = &op.getElseRegion().front();
    mlir::Block *elseLast = &op.getElseRegion().back();

    rewriter.inlineRegionBefore(op.getElseRegion(), continuation);

    rewriter.create<mlir::cf::CondBranchOp>(op->getLoc(), conditionValue,
                                            thenFirst, std::nullopt, elseFirst,
                                            std::nullopt);

    // Branch to the continuation block.
    rewriter.setInsertionPointToEnd(thenLast);
    rewriter.create<mlir::cf::BranchOp>(op->getLoc(), continuation);

    rewriter.setInsertionPointToEnd(elseLast);
    rewriter.create<mlir::cf::BranchOp>(op->getLoc(), continuation);

    // Erase the operation.
    rewriter.eraseOp(op);

    if (mlir::failed(recurse(rewriter, elseFirst, elseLast, loopExitBlock,
                             functionReturnBlock))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult
BaseModelicaToCFConversionPass::createCFG(mlir::RewriterBase &rewriter,
                                          WhileOp op,
                                          mlir::Block *functionReturnBlock) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  // Split the current block.
  mlir::Block *currentBlock = rewriter.getInsertionBlock();

  mlir::Block *continuation =
      rewriter.splitBlock(currentBlock, op->getIterator());

  // Keep the references to the op blocks.
  mlir::Block *conditionFirst = &op.getConditionRegion().front();
  mlir::Block *conditionLast = &op.getConditionRegion().back();

  mlir::Block *bodyFirst = &op.getBodyRegion().front();
  mlir::Block *bodyLast = &op.getBodyRegion().back();

  // Inline the regions.
  rewriter.inlineRegionBefore(op.getConditionRegion(), continuation);
  rewriter.inlineRegionBefore(op.getBodyRegion(), continuation);

  // Branch to the "condition" region.
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<mlir::cf::BranchOp>(op->getLoc(), conditionFirst);

  // Branch to the "body" region.
  rewriter.setInsertionPointToEnd(conditionLast);

  auto conditionOp = mlir::cast<ConditionOp>(conditionLast->getTerminator());

  mlir::Value conditionValue = conditionOp.getCondition();

  if (conditionValue.getType() != rewriter.getI1Type()) {
    conditionValue = rewriter.create<CastOp>(
        conditionOp.getLoc(), rewriter.getI1Type(), conditionValue);
  }

  rewriter.create<mlir::cf::CondBranchOp>(op->getLoc(), conditionValue,
                                          bodyFirst, std::nullopt, continuation,
                                          std::nullopt);

  rewriter.eraseOp(conditionOp);

  // Branch back to the "condition" region.
  rewriter.setInsertionPointToEnd(bodyLast);
  rewriter.create<mlir::cf::BranchOp>(op->getLoc(), conditionFirst);

  // Erase the operation.
  rewriter.eraseOp(op);

  // Recurse on the body operations.
  return recurse(rewriter, bodyFirst, bodyLast, continuation,
                 functionReturnBlock);
}

mlir::LogicalResult
BaseModelicaToCFConversionPass::createCFG(mlir::RewriterBase &rewriter,
                                          ReturnOp op,
                                          mlir::Block *functionReturnBlock) {
  if (!functionReturnBlock) {
    return mlir::failure();
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  mlir::Block *currentBlock = rewriter.getInsertionBlock();
  rewriter.splitBlock(currentBlock, op->getIterator());

  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<mlir::cf::BranchOp>(op->getLoc(), functionReturnBlock);

  rewriter.eraseOp(op);
  return mlir::success();
}

mlir::LogicalResult BaseModelicaToCFConversionPass::recurse(
    mlir::RewriterBase &rewriter, mlir::Block *first, mlir::Block *last,
    mlir::Block *loopExitBlock, mlir::Block *functionReturnBlock) {
  llvm::SmallVector<mlir::Operation *> ops;
  auto it = first->getIterator();

  do {
    for (auto &op : it->getOperations()) {
      ops.push_back(&op);
    }
  } while (it++ != last->getIterator());

  for (auto &op : ops) {
    if (mlir::failed(
            createCFG(rewriter, op, loopExitBlock, functionReturnBlock))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

namespace mlir {
std::unique_ptr<mlir::Pass> createBaseModelicaToCFConversionPass() {
  return std::make_unique<BaseModelicaToCFConversionPass>();
}
} // namespace mlir
