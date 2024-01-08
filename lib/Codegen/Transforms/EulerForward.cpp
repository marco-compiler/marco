#include "marco/Codegen/Transforms/EulerForward.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "marco/Codegen/Transforms/Solvers/Common.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "euler-forward"

namespace mlir::modelica
{
#define GEN_PASS_DEF_EULERFORWARDPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class EulerForwardPass
      : public mlir::modelica::impl::EulerForwardPassBase<EulerForwardPass>,
        public mlir::modelica::impl::ModelSolver
  {
    public:
      using EulerForwardPassBase<EulerForwardPass>::EulerForwardPassBase;

      void runOnOperation() override;

    private:
      DerivativesMap& getDerivativesMap(ModelOp modelOp);

      mlir::LogicalResult processModelOp(ModelOp modelOp);

      mlir::LogicalResult solveICModel(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          llvm::ArrayRef<SCCOp> SCCs);

      mlir::LogicalResult solveMainModel(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variables,
          llvm::ArrayRef<SCCOp> SCCs);

      mlir::LogicalResult createCalcICFunction(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<SCCOp> SCCs);

      mlir::LogicalResult createUpdateNonStateVariablesFunction(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<SCCOp> SCCs);

      mlir::LogicalResult createUpdateStateVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variableOps);
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

  llvm::SmallVector<SCCOp> initialSCCs;
  llvm::SmallVector<SCCOp> mainSCCs;

  modelOp.collectInitialSCCs(initialSCCs);
  modelOp.collectMainSCCs(mainSCCs);

  llvm::SmallVector<VariableOp> variables;
  modelOp.collectVariables(variables);

  // Solve the 'initial conditions' model.
  if (mlir::failed(solveICModel(
          rewriter, symbolTableCollection, modelOp, initialSCCs))) {
    return mlir::failure();
  }

  // Solve the 'main' model.
  if (mlir::failed(solveMainModel(
          rewriter, symbolTableCollection, modelOp, variables, mainSCCs))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult EulerForwardPass::solveICModel(
    mlir::IRRewriter& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    llvm::ArrayRef<SCCOp> SCCs)
{
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

  if (mlir::failed(createCalcICFunction(
          rewriter, symbolTableCollection, moduleOp, modelOp, SCCs))) {
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
          rewriter, symbolTableCollection, moduleOp, modelOp, SCCs))) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateStateVariablesFunction(
          rewriter, symbolTableCollection, moduleOp, modelOp, variables))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult EulerForwardPass::createCalcICFunction(
    mlir::IRRewriter& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<SCCOp> SCCs)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Create the function.
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::simulation::FunctionOp>(
      modelOp.getLoc(), "calcIC",
      rewriter.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  if (!SCCs.empty()) {
    // Create the schedule operation.
    ScheduleOp scheduleOp = createSchedule(
        rewriter, symbolTableCollection, moduleOp, modelOp.getLoc(),
        "ic_equations", SCCs);

    if (!scheduleOp) {
      return mlir::failure();
    }

    rewriter.create<RunScheduleOp>(
        modelOp.getLoc(), mlir::SymbolRefAttr::get(scheduleOp.getSymNameAttr()));
  }

  rewriter.create<mlir::simulation::ReturnOp>(modelOp.getLoc(), std::nullopt);
  return mlir::success();
}

mlir::LogicalResult EulerForwardPass::createUpdateNonStateVariablesFunction(
    mlir::IRRewriter& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<SCCOp> SCCs)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Create the function.
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::simulation::FunctionOp>(
      modelOp.getLoc(), "updateNonStateVariables",
      rewriter.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  if (!SCCs.empty()) {
    // Create the schedule operation.
    ScheduleOp scheduleOp = createSchedule(
        rewriter, symbolTableCollection, moduleOp, modelOp.getLoc(),
        "main_equations", SCCs);

    if (!scheduleOp) {
      return mlir::failure();
    }

    rewriter.create<RunScheduleOp>(
        modelOp.getLoc(), mlir::SymbolRefAttr::get(scheduleOp.getSymNameAttr()));
  }

  rewriter.create<mlir::simulation::ReturnOp>(modelOp.getLoc(), std::nullopt);
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

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      modelOp.getLoc(), "updateStateVariables",
      builder.getFunctionType(builder.getF64Type(), std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  mlir::Value timeStep = functionOp.getArgument(0);

  auto apply = [&](mlir::OpBuilder& nestedBuilder,
                   mlir::Value scalarState,
                   mlir::Value scalarDerivative) -> mlir::Value {
    mlir::Value result = nestedBuilder.create<MulOp>(
        modelOp.getLoc(), scalarDerivative.getType(), scalarDerivative,
        timeStep);

    result = nestedBuilder.create<AddOp>(
        modelOp.getLoc(), scalarState.getType(), scalarState, result);

    return result;
  };

  auto& derivativesMap = getDerivativesMap(modelOp);

  for (VariableOp variableOp : variableOps) {
    if (auto derivativeName = derivativesMap.getDerivative(
            mlir::FlatSymbolRefAttr::get(variableOp.getSymNameAttr()))) {
      auto stateSimulationVarOp =
          symbolTableCollection.lookupSymbolIn<SimulationVariableOp>(
              moduleOp, variableOp.getSymNameAttr());

      if (!stateSimulationVarOp) {
        variableOp.emitError() << "simulation state variable not found";
        return mlir::failure();
      }

      assert(derivativeName->getNestedReferences().empty());

      auto derivativeSimulationVarOp =
          symbolTableCollection.lookupSymbolIn<SimulationVariableOp>(
              moduleOp, derivativeName->getRootReference());

      if (!derivativeSimulationVarOp) {
        variableOp.emitError() << "derivative state variable not found";
        return mlir::failure();
      }

      VariableType variableType = variableOp.getVariableType();

      if (variableType.isScalar()) {
        mlir::Value stateValue = builder.create<SimulationVariableGetOp>(
            stateSimulationVarOp.getLoc(), stateSimulationVarOp);

        mlir::Value derivativeValue = builder.create<SimulationVariableGetOp>(
            derivativeSimulationVarOp.getLoc(), derivativeSimulationVarOp);

        mlir::Value updatedValue = apply(builder, stateValue, derivativeValue);

        builder.create<SimulationVariableSetOp>(
            stateSimulationVarOp.getLoc(), stateSimulationVarOp, updatedValue);
      } else {

        mlir::Value state = builder.create<SimulationVariableGetOp>(
            variableOp.getLoc(), stateSimulationVarOp);

        mlir::Value derivative = builder.create<SimulationVariableGetOp>(
            variableOp.getLoc(), derivativeSimulationVarOp);

        // Create the loops to iterate on each scalar variable.
        llvm::SmallVector<mlir::Value, 3> lowerBounds;
        llvm::SmallVector<mlir::Value, 3> upperBounds;
        llvm::SmallVector<mlir::Value, 3> steps;

        for (unsigned int i = 0; i < variableOp.getVariableType().getRank(); ++i) {
          lowerBounds.push_back(builder.create<ConstantOp>(
              modelOp.getLoc(), builder.getIndexAttr(0)));

          mlir::Value dimension = builder.create<ConstantOp>(
              modelOp.getLoc(), builder.getIndexAttr(i));

          upperBounds.push_back(builder.create<DimOp>(
              modelOp.getLoc(), state, dimension));

          steps.push_back(builder.create<ConstantOp>(
              modelOp.getLoc(), builder.getIndexAttr(1)));
        }

        mlir::scf::buildLoopNest(
            builder, modelOp.getLoc(), lowerBounds, upperBounds, steps,
            [&](mlir::OpBuilder& nestedBuilder,
                mlir::Location loc,
                mlir::ValueRange indices) {
              mlir::Value scalarState = nestedBuilder.create<LoadOp>(
                  loc, state, indices);

              mlir::Value scalarDerivative = nestedBuilder.create<LoadOp>(
                  loc, derivative, indices);

              mlir::Value updatedValue = apply(
                  nestedBuilder, scalarState, scalarDerivative);

              nestedBuilder.create<StoreOp>(loc, updatedValue, state, indices);
            });
      }
    }
  }

  // Terminate the function.
  builder.setInsertionPointToEnd(entryBlock);
  builder.create<mlir::simulation::ReturnOp>(modelOp.getLoc(), std::nullopt);

  return mlir::success();
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createEulerForwardPass()
  {
    return std::make_unique<EulerForwardPass>();
  }
}
