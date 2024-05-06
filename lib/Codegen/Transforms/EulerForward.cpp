#include "marco/Codegen/Transforms/EulerForward.h"
#include "marco/Dialect/Runtime/RuntimeDialect.h"
#include "marco/Codegen/Analysis/DerivativesMap.h"
#include "marco/Codegen/Conversion/BaseModelicaCommon/TypeConverter.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "euler-forward"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_EULERFORWARDPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
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
      DerivativesMap& getDerivativesMap(ModelOp modelOp);

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

      mlir::LogicalResult cleanModelOp(ModelOp modelOp);
  };
}

void EulerForwardPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  llvm::SmallVector<ModelOp, 1> modelOps;

  for (ModelOp modelOp : moduleOp.getOps<ModelOp>()) {
    modelOps.push_back(modelOp);
  }

  for (ModelOp modelOp : modelOps) {
    if (mlir::failed(processModelOp(modelOp))) {
      return signalPassFailure();
    }

    if (mlir::failed(cleanModelOp(modelOp))) {
      return signalPassFailure();
    }
  }

  markAnalysesPreserved<DerivativesMap>();
}

DerivativesMap& EulerForwardPass::getDerivativesMap(ModelOp modelOp)
{
  if (auto analysis = getCachedChildAnalysis<DerivativesMap>(modelOp)) {
    return *analysis;
  }

  auto& analysis = getChildAnalysis<DerivativesMap>(modelOp);
  analysis.initialize();
  return analysis;
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

static void createStateUpdateFunctionCall(
    mlir::OpBuilder& builder,
    VariableOp stateVariable,
    VariableOp derivativeVariable,
    MultidimensionalRangeAttr ranges,
    EquationFunctionOp equationFuncOp)
{
  llvm::SmallVector<mlir::Attribute> writtenVarAttrs;
  llvm::SmallVector<mlir::Attribute> readVarAttrs;

  IndexSet variableIndices;

  if (ranges) {
    variableIndices += ranges.getValue();
  }

  writtenVarAttrs.push_back(VariableAttr::get(
      builder.getContext(),
      mlir::SymbolRefAttr::get(stateVariable.getSymNameAttr()),
      IndexSetAttr::get(builder.getContext(), variableIndices)));

  readVarAttrs.push_back(VariableAttr::get(
      builder.getContext(),
      mlir::SymbolRefAttr::get(derivativeVariable.getSymNameAttr()),
      IndexSetAttr::get(builder.getContext(), variableIndices)));

  auto blockOp = builder.create<ScheduleBlockOp>(
      stateVariable.getLoc(),
      true,
      builder.getArrayAttr(writtenVarAttrs),
      builder.getArrayAttr(readVarAttrs));

  builder.createBlock(&blockOp.getBodyRegion());
  builder.setInsertionPointToStart(blockOp.getBody());

  builder.create<EquationCallOp>(
      equationFuncOp.getLoc(), equationFuncOp.getSymName(),
      ranges, true);
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
  auto& derivativesMap = getDerivativesMap(modelOp);

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

  auto getNewStateValueFn =
      [&](mlir::OpBuilder& nestedBuilder,
          mlir::Value scalarState,
          mlir::Value scalarDerivative,
          mlir::Value timeStep) -> mlir::Value {
    mlir::Value result = nestedBuilder.create<MulOp>(
        modelOp.getLoc(), scalarDerivative.getType(), scalarDerivative,
        timeStep);

    result = nestedBuilder.create<AddOp>(
        modelOp.getLoc(), scalarState.getType(), scalarState, result);

    return result;
  };

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
      // Create the equation function.
      builder.setInsertionPointToEnd(moduleOp.getBody());
      auto variableType = stateVariable.getVariableType();

      auto equationFuncOp = builder.create<EquationFunctionOp>(
          stateVariable.getLoc(), "euler_state_update",
          variableType.getRank());

      symbolTableCollection.getSymbolTable(moduleOp).insert(equationFuncOp);
      mlir::Block* equationFuncBody = equationFuncOp.addEntryBlock();
      builder.setInsertionPointToStart(equationFuncBody);

      mlir::Value timeStep = builder.create<GlobalVariableGetOp>(
          stateVariable.getLoc(), timeStepVariable);

      timeStep = builder.create<LoadOp>(
          timeStep.getLoc(), timeStep, std::nullopt);

      mlir::Value state = builder.create<QualifiedVariableGetOp>(
          equationFuncOp.getLoc(), stateVariable);

      mlir::Value derivative = builder.create<QualifiedVariableGetOp>(
          equationFuncOp.getLoc(), derivativeVariable);

      if (stateVariable.getVariableType().isScalar()) {
        // Scalar variable.
        mlir::Value newStateValue =
            getNewStateValueFn(builder, state, derivative, timeStep);

        builder.create<QualifiedVariableSetOp>(
            equationFuncOp.getLoc(), stateVariable, newStateValue);
      } else {
        // Array variable.
        mlir::Value step = builder.create<mlir::arith::ConstantOp>(
            equationFuncOp.getLoc(), builder.getIndexAttr(1));

        llvm::SmallVector<mlir::Value, 3> inductions;

        for (int64_t i = 0, e = variableType.getRank(); i < e; ++i) {
          auto forOp = builder.create<mlir::scf::ForOp>(
              equationFuncOp.getLoc(),
              equationFuncOp.getLowerBound(i),
              equationFuncOp.getUpperBound(i),
              step);

          inductions.push_back(forOp.getInductionVar());
          builder.setInsertionPointToStart(forOp.getBody());
        }

        mlir::Value scalarState = builder.create<LoadOp>(
            state.getLoc(), state, inductions);

        mlir::Value scalarDerivative = builder.create<LoadOp>(
            derivative.getLoc(), derivative, inductions);

        mlir::Value newScalarStateValue = getNewStateValueFn(
            builder, scalarState, scalarDerivative, timeStep);

        builder.create<StoreOp>(
            state.getLoc(), newScalarStateValue, state, inductions);
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
    }

    // Run the schedule.
    builder.setInsertionPointToEnd(functionBody);
    builder.create<RunScheduleOp>(scheduleOp.getLoc(), scheduleOp);
  }

  // Terminate the function.
  builder.create<mlir::runtime::ReturnOp>(modelOp.getLoc(), std::nullopt);

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
}
