#include "marco/Dialect/BaseModelica/Transforms/EulerForward.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/Runtime/IR/Runtime.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "euler-forward"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_EULERFORWARDPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class EulerForwardPass
      : public mlir::bmodelica::impl::EulerForwardPassBase<EulerForwardPass>
  {
    public:
      using EulerForwardPassBase<EulerForwardPass>::EulerForwardPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult processModelOp(ModelOp modelOp);

      mlir::LogicalResult solveMainModel(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variables,
          llvm::ArrayRef<SCCOp> SCCs);

      mlir::LogicalResult createUpdateNonStateVariablesFunction(
          mlir::IRRewriter& rewriter,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<SCCOp> SCCs);

      mlir::LogicalResult createUpdateStateVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variableOps);

      mlir::LogicalResult createRangedStateVariableUpdateBlocks(
          mlir::OpBuilder& builder,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          DynamicOp dynamicOp,
          VariableOp stateVariable,
          VariableOp derivativeVariable,
          GlobalVariableOp timeStepVariable);

      mlir::LogicalResult createMonolithicStateVariableUpdateBlock(
          mlir::OpBuilder& builder,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          DynamicOp dynamicOp,
          VariableOp stateVariable,
          VariableOp derivativeVariable,
          GlobalVariableOp timeStepVariable);

      mlir::LogicalResult cleanModelOp(ModelOp modelOp);
  };
}

void EulerForwardPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation* op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  for (ModelOp modelOp : modelOps) {
    if (mlir::failed(processModelOp(modelOp))) {
      return signalPassFailure();
    }

    if (mlir::failed(cleanModelOp(modelOp))) {
      return signalPassFailure();
    }
  }
}

mlir::LogicalResult EulerForwardPass::processModelOp(ModelOp modelOp)
{
  mlir::IRRewriter rewriter(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;

  llvm::SmallVector<SCCOp> mainSCCs;
  modelOp.collectMainSCCs(mainSCCs);

  llvm::SmallVector<VariableOp> variables;
  modelOp.collectVariables(variables);

  // Solve the 'main' model.
  if (mlir::failed(solveMainModel(
          rewriter, symbolTableCollection, modelOp, variables, mainSCCs))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult EulerForwardPass::solveMainModel(
    mlir::IRRewriter& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variables,
    llvm::ArrayRef<SCCOp> SCCs)
{
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

  if (mlir::failed(createUpdateNonStateVariablesFunction(
          rewriter, moduleOp, modelOp, SCCs))) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateStateVariablesFunction(
          rewriter, symbolTableCollection, moduleOp, modelOp, variables))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult EulerForwardPass::createUpdateNonStateVariablesFunction(
    mlir::IRRewriter& rewriter,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<SCCOp> SCCs)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Create the function running the schedule.
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::runtime::FunctionOp>(
      modelOp.getLoc(), "updateNonStateVariables",
      rewriter.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  if (!SCCs.empty()) {
    rewriter.setInsertionPointToEnd(modelOp.getBody());

    // Create the schedule operation.
    auto scheduleOp = rewriter.create<ScheduleOp>(modelOp.getLoc(), "dynamic");
    rewriter.createBlock(&scheduleOp.getBodyRegion());
    rewriter.setInsertionPointToStart(scheduleOp.getBody());

    auto dynamicOp = rewriter.create<DynamicOp>(modelOp.getLoc());
    rewriter.createBlock(&dynamicOp.getBodyRegion());
    rewriter.setInsertionPointToStart(dynamicOp.getBody());

    for (SCCOp scc : SCCs) {
      scc->moveBefore(dynamicOp.getBody(), dynamicOp.getBody()->end());
    }

    // Call the schedule.
    rewriter.setInsertionPointToEnd(entryBlock);
    rewriter.create<RunScheduleOp>(modelOp.getLoc(), scheduleOp);
  }

  rewriter.setInsertionPointToEnd(entryBlock);
  rewriter.create<mlir::runtime::ReturnOp>(modelOp.getLoc(), std::nullopt);
  return mlir::success();
}

mlir::LogicalResult EulerForwardPass::createUpdateStateVariablesFunction(
    mlir::OpBuilder& builder,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  // Create a global variable storing the value of the requested time step.
  auto timeStepVariable = builder.create<GlobalVariableOp>(
      modelOp.getLoc(), "timeStep",
      ArrayType::get(std::nullopt, builder.getF64Type()));

  symbolTableCollection.getSymbolTable(moduleOp).insert(timeStepVariable);

  // Create the function called by the runtime library.
  auto functionOp = builder.create<mlir::runtime::FunctionOp>(
      modelOp.getLoc(), "updateStateVariables",
      builder.getFunctionType(builder.getF64Type(), std::nullopt));

  mlir::Block* functionBody = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(functionBody);

  mlir::Value timeStepArg = functionOp.getArgument(0);

  // Set the time step in the global variable.
  auto timeStepArray = builder.create<GlobalVariableGetOp>(
      timeStepArg.getLoc(), timeStepVariable);

  builder.create<StoreOp>(
      timeStepArg.getLoc(), timeStepArg, timeStepArray, std::nullopt);

  // Compute the list of state and derivative variables.
  const DerivativesMap& derivativesMap =
      modelOp.getProperties().derivativesMap;

  // The two lists are kept in sync.
  llvm::SmallVector<VariableOp> stateVariables;
  llvm::SmallVector<VariableOp> derivativeVariables;

  for (VariableOp variableOp : variableOps) {
    if (auto derivativeName = derivativesMap.getDerivative(
        mlir::FlatSymbolRefAttr::get(variableOp.getSymNameAttr()))) {
      auto derivativeVariableOp =
          symbolTableCollection.lookupSymbolIn<VariableOp>(
              modelOp, *derivativeName);

      stateVariables.push_back(variableOp);
      derivativeVariables.push_back(derivativeVariableOp);
    }
  }

  assert(stateVariables.size() == derivativeVariables.size());

  if (!stateVariables.empty()) {
    // Create the schedule.
    builder.setInsertionPointToEnd(modelOp.getBody());

    auto scheduleOp = builder.create<ScheduleOp>(
        modelOp.getLoc(), "schedule_state_variables");

    symbolTableCollection.getSymbolTable(modelOp).insert(scheduleOp);

    builder.createBlock(&scheduleOp.getBodyRegion());
    builder.setInsertionPointToStart(scheduleOp.getBody());

    auto dynamicOp = builder.create<DynamicOp>(scheduleOp.getLoc());
    builder.createBlock(&dynamicOp.getBodyRegion());
    builder.setInsertionPointToStart(dynamicOp.getBody());

    for (const auto& [stateVariable, derivativeVariable] :
         llvm::zip(stateVariables, derivativeVariables)) {
      if (rangedStateUpdateFunctions) {
        if (mlir::failed(createRangedStateVariableUpdateBlocks(
                builder, symbolTableCollection, moduleOp, dynamicOp,
                stateVariable, derivativeVariable, timeStepVariable))) {
          return mlir::failure();
        }
      } else {
        if (mlir::failed(createMonolithicStateVariableUpdateBlock(
                builder, symbolTableCollection, moduleOp, dynamicOp,
                stateVariable, derivativeVariable, timeStepVariable))) {
          return mlir::failure();
        }
      }
    }

    // Run the schedule.
    builder.setInsertionPointToEnd(functionBody);
    builder.create<RunScheduleOp>(scheduleOp.getLoc(), scheduleOp);
  }

  // Terminate the function.
  builder.create<mlir::runtime::ReturnOp>(modelOp.getLoc(), std::nullopt);

  return mlir::success();
}

static void getStateUpdateBlockVarWriteReadInfo(
    mlir::OpBuilder& builder,
    VariableOp stateVariable,
    VariableOp derivativeVariable,
    MultidimensionalRangeAttr ranges,
    llvm::SmallVectorImpl<Variable>& writtenVariables,
    llvm::SmallVectorImpl<Variable>& readVariables)
{
  IndexSet indices;

  if (ranges) {
    indices += ranges.getValue();
  }

  writtenVariables.emplace_back(
      mlir::SymbolRefAttr::get(stateVariable.getSymNameAttr()),
      indices);

  readVariables.emplace_back(
      mlir::SymbolRefAttr::get(derivativeVariable.getSymNameAttr()),
      indices);
}

static void createStateUpdateFunctionCall(
    mlir::OpBuilder& builder,
    VariableOp stateVariable,
    VariableOp derivativeVariable,
    MultidimensionalRangeAttr ranges,
    EquationFunctionOp equationFuncOp)
{
  auto blockOp = builder.create<ScheduleBlockOp>(
      stateVariable.getLoc(), true);

  getStateUpdateBlockVarWriteReadInfo(
      builder, stateVariable, derivativeVariable, ranges,
      blockOp.getProperties().writtenVariables,
      blockOp.getProperties().readVariables);

  builder.createBlock(&blockOp.getBodyRegion());
  builder.setInsertionPointToStart(blockOp.getBody());

  builder.create<EquationCallOp>(
      equationFuncOp.getLoc(), equationFuncOp.getSymName(),
      ranges, true);
}

mlir::LogicalResult EulerForwardPass::createRangedStateVariableUpdateBlocks(
    mlir::OpBuilder& builder,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    DynamicOp dynamicOp,
    VariableOp stateVariable,
    VariableOp derivativeVariable,
    GlobalVariableOp timeStepVariable)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  // Create the equation function.
  builder.setInsertionPointToEnd(moduleOp.getBody());
  auto variableType = stateVariable.getVariableType();
  int64_t variableRank = variableType.getRank();

  auto equationFuncOp = builder.create<EquationFunctionOp>(
      stateVariable.getLoc(),
      "euler_state_update_" + stateVariable.getSymName().str(),
      variableRank);

  symbolTableCollection.getSymbolTable(moduleOp).insert(equationFuncOp);
  mlir::Block* equationFuncBody = equationFuncOp.addEntryBlock();
  builder.setInsertionPointToStart(equationFuncBody);

  mlir::Value timeStep = builder.create<GlobalVariableGetOp>(
      stateVariable.getLoc(), timeStepVariable);

  timeStep = builder.create<LoadOp>(
      timeStep.getLoc(), timeStep, std::nullopt);

  auto getNewScalarStateFn =
      [&](mlir::OpBuilder& nestedBuilder,
          mlir::Location nestedLoc,
          mlir::Value scalarState,
          mlir::Value scalarDerivative) -> mlir::Value {
    mlir::Value result = nestedBuilder.create<MulOp>(
        nestedLoc, scalarDerivative.getType(), scalarDerivative,
        timeStep);

    result = nestedBuilder.create<AddOp>(
        nestedLoc, scalarState.getType(), scalarState, result);

    return result;
  };

  if (variableRank == 0) {
    // Scalar variable.
    mlir::Value state = builder.create<QualifiedVariableGetOp>(
        equationFuncOp.getLoc(), stateVariable);

    mlir::Value derivative = builder.create<QualifiedVariableGetOp>(
        equationFuncOp.getLoc(), derivativeVariable);

    mlir::Value updatedState = getNewScalarStateFn(
        builder, equationFuncOp.getLoc(), state, derivative);

    builder.create<QualifiedVariableSetOp>(
        equationFuncOp.getLoc(), stateVariable, updatedState);
  } else {
    // Array variable.
    mlir::Value state = builder.create<QualifiedVariableGetOp>(
        equationFuncOp.getLoc(),
        stateVariable.getVariableType().toArrayType(),
        getSymbolRefFromRoot(stateVariable));

    mlir::Value derivative = builder.create<QualifiedVariableGetOp>(
        equationFuncOp.getLoc(),
        derivativeVariable.getVariableType().toArrayType(),
        getSymbolRefFromRoot(derivativeVariable));

    llvm::SmallVector<mlir::Value> lowerBounds;
    llvm::SmallVector<mlir::Value> upperBounds;
    llvm::SmallVector<int64_t> steps(variableRank, 1);

    for (int64_t dim = 0; dim < variableRank; ++dim) {
      lowerBounds.push_back(equationFuncOp.getLowerBound(dim));
      upperBounds.push_back(equationFuncOp.getUpperBound(dim));
    }

    mlir::affine::buildAffineLoopNest(
        builder, equationFuncOp.getLoc(), lowerBounds, upperBounds, steps,
        [&](mlir::OpBuilder& nestedBuilder, mlir::Location nestedLoc,
            mlir::ValueRange indices) {
          mlir::Value scalarState = nestedBuilder.create<LoadOp>(
              nestedLoc, state, indices);

          mlir::Value scalarDerivative = nestedBuilder.create<LoadOp>(
              nestedLoc, derivative, indices);

          mlir::Value updatedScalarState = getNewScalarStateFn(
              nestedBuilder, nestedLoc, scalarState, scalarDerivative);

          nestedBuilder.create<StoreOp>(
              nestedLoc, updatedScalarState, state, indices);
        });
  }

  builder.setInsertionPointToEnd(equationFuncBody);
  builder.create<YieldOp>(equationFuncOp.getLoc());

  // Create the schedule blocks and the calls to the equation function.
  builder.setInsertionPointToEnd(dynamicOp.getBody());

  IndexSet indices =
      stateVariable.getIndices().getCanonicalRepresentation();

  if (indices.empty()) {
    createStateUpdateFunctionCall(
        builder, stateVariable, derivativeVariable, nullptr,
        equationFuncOp);
  } else {
    for (const MultidimensionalRange& range : llvm::make_range(
             indices.rangesBegin(), indices.rangesEnd())) {

      createStateUpdateFunctionCall(
          builder, stateVariable, derivativeVariable,
          MultidimensionalRangeAttr::get(&getContext(), range),
          equationFuncOp);
    }
  }

  return mlir::success();
}

mlir::LogicalResult EulerForwardPass::createMonolithicStateVariableUpdateBlock(
    mlir::OpBuilder& builder,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    DynamicOp dynamicOp,
    VariableOp stateVariable,
    VariableOp derivativeVariable,
    GlobalVariableOp timeStepVariable)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  // Create the equation function.
  builder.setInsertionPointToEnd(moduleOp.getBody());
  auto variableType = stateVariable.getVariableType();

  auto funcOp = builder.create<RawFunctionOp>(
      stateVariable.getLoc(),
      "euler_state_update_" + stateVariable.getSymName().str(),
      builder.getFunctionType(std::nullopt, std::nullopt));

  symbolTableCollection.getSymbolTable(moduleOp).insert(funcOp);

  mlir::Block* funcBody = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(funcBody);

  mlir::Value timeStep = builder.create<GlobalVariableGetOp>(
      stateVariable.getLoc(), timeStepVariable);

  timeStep = builder.create<LoadOp>(
      timeStep.getLoc(), timeStep, std::nullopt);

  mlir::Value state = builder.create<QualifiedVariableGetOp>(
      funcOp.getLoc(), stateVariable);

  mlir::Value derivative = builder.create<QualifiedVariableGetOp>(
      funcOp.getLoc(), derivativeVariable);

  mlir::Value mulOp = builder.create<MulOp>(
      funcOp.getLoc(), timeStep, derivative);

  mlir::Value addOp = builder.create<AddOp>(
      funcOp.getLoc(), state, mulOp);

  builder.create<QualifiedVariableSetOp>(
      funcOp.getLoc(), stateVariable, addOp);

  builder.setInsertionPointToEnd(funcBody);
  builder.create<RawReturnOp>(funcOp.getLoc());

  // Create the schedule block and call the function.
  builder.setInsertionPointToEnd(dynamicOp.getBody());

  int64_t variableRank = variableType.getRank();

  auto blockOp = builder.create<ScheduleBlockOp>(
      stateVariable.getLoc(), true);

  if (variableRank == 0) {
    getStateUpdateBlockVarWriteReadInfo(
        builder, stateVariable, derivativeVariable, nullptr,
        blockOp.getProperties().writtenVariables,
        blockOp.getProperties().readVariables);
  } else {
    for (int64_t dim = 0; dim < variableType.getRank(); ++dim) {
      llvm::SmallVector<Range> ranges;

      for (int64_t dimSize : variableType.getShape()) {
        assert(dimSize != mlir::ShapedType::kDynamic);
        ranges.push_back(Range(0, dimSize));
      }

      getStateUpdateBlockVarWriteReadInfo(
          builder, stateVariable, derivativeVariable,
          MultidimensionalRangeAttr::get(
              builder.getContext(), MultidimensionalRange(ranges)),
          blockOp.getProperties().writtenVariables,
          blockOp.getProperties().readVariables);
    }
  }

  builder.createBlock(&blockOp.getBodyRegion());
  builder.setInsertionPointToStart(blockOp.getBody());
  builder.create<CallOp>(funcOp.getLoc(), funcOp);

  return mlir::success();
}

mlir::LogicalResult EulerForwardPass::cleanModelOp(ModelOp modelOp)
{
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  return mlir::applyPatternsAndFoldGreedily(modelOp, std::move(patterns));
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createEulerForwardPass()
  {
    return std::make_unique<EulerForwardPass>();
  }

  std::unique_ptr<mlir::Pass> createEulerForwardPass(
      const EulerForwardPassOptions& options)
  {
    return std::make_unique<EulerForwardPass>(options);
  }
}
