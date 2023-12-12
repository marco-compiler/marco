#include "marco/Codegen/Transforms/EulerForward.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "marco/Codegen/Transforms/SolverPassBase.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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
      using EulerForwardPassBase::EulerForwardPassBase;

      void runOnOperation() override;

    private:
      DerivativesMap& getDerivativesMap(ModelOp modelOp);

    protected:
      mlir::LogicalResult solveICModel(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variableOps,
          const DerivativesMap& derivativesMap,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          llvm::ArrayRef<SCCOp> SCCs) override;

      mlir::LogicalResult solveMainModel(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variableOps,
          const DerivativesMap& derivativesMap,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          llvm::ArrayRef<SCCOp> SCCs) override;

    private:
      mlir::LogicalResult createCalcICFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          llvm::ArrayRef<SCCOp> SCCs,
          const llvm::DenseMap<
              ScheduledEquationInstanceOp, RawFunctionOp>& equationFunctions);

      mlir::LogicalResult createUpdateNonStateVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          llvm::ArrayRef<SCCOp> SCCs,
          const llvm::DenseMap<
              ScheduledEquationInstanceOp, RawFunctionOp>& equationFunctions);

      mlir::LogicalResult createUpdateStateVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          mlir::SymbolTableCollection& symbolTableCollection,
          llvm::ArrayRef<VariableOp> variableOps,
          const DerivativesMap& derivativesMap,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap);
  };
}

void EulerForwardPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  llvm::SmallVector<ModelOp> modelOps;

  for (ModelOp modelOp : moduleOp.getOps<ModelOp>()) {
    modelOps.push_back(modelOp);
  }

  auto expectedVariablesFilter =
      marco::VariableFilter::fromString(variablesFilter);

  std::unique_ptr<marco::VariableFilter> variablesFilterInstance;

  if (!expectedVariablesFilter) {
    getOperation().emitWarning()
        << "Invalid variable filter string. No filtering will take place";

    variablesFilterInstance = std::make_unique<marco::VariableFilter>();
  } else {
    variablesFilterInstance = std::make_unique<marco::VariableFilter>(
        std::move(*expectedVariablesFilter));
  }

  for (ModelOp modelOp : modelOps) {
    if (mlir::failed(convert(
            modelOp, getDerivativesMap(modelOp), *variablesFilterInstance,
            processICModel, processMainModel))) {
      return signalPassFailure();
    }
  }
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

mlir::LogicalResult EulerForwardPass::solveICModel(
    mlir::IRRewriter& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps,
    const DerivativesMap& derivativesMap,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    llvm::ArrayRef<SCCOp> SCCs)
{
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();
  llvm::DenseSet<ScheduledEquationInstanceOp> explicitEquations;

  for (SCCOp scc : SCCs) {
    if (scc.getCycle()) {
      return mlir::failure();
    }

    llvm::SmallVector<ScheduledEquationInstanceOp> equationOps;
    scc.collectEquations(equationOps);

    for (ScheduledEquationInstanceOp equationOp : equationOps) {
      auto explicitEquationOp = equationOp.cloneAndExplicitate(
          rewriter, symbolTableCollection);

      if (!explicitEquationOp) {
        equationOp.cloneAndExplicitate(
            rewriter, symbolTableCollection);

        return mlir::failure();
      }

      rewriter.eraseOp(equationOp);
      explicitEquations.insert(explicitEquationOp);
    }
  }

  // Convert the explicit equations into functions.
  llvm::DenseMap<ScheduledEquationInstanceOp, RawFunctionOp> equationFunctions;
  size_t equationFunctionsCounter = 0;

  for (ScheduledEquationInstanceOp equationOp : explicitEquations) {
    RawFunctionOp templateFunction = createEquationTemplateFunction(
        rewriter, moduleOp, symbolTableCollection,
        equationOp.getTemplate(),
        equationOp.getViewElementIndex(),
        equationOp.getIterationDirections(),
        "initial_equation_" + std::to_string(equationFunctionsCounter++),
        localToGlobalVariablesMap);

    if (!templateFunction) {
      return mlir::failure();
    }

    equationFunctions[equationOp] = templateFunction;
  }

  if (mlir::failed(createCalcICFunction(
          rewriter, moduleOp, modelOp.getLoc(), SCCs, equationFunctions))) {
      return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult EulerForwardPass::solveMainModel(
    mlir::IRRewriter& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps,
    const DerivativesMap& derivativesMap,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    llvm::ArrayRef<SCCOp> SCCs)
{
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();
  llvm::DenseSet<ScheduledEquationInstanceOp> explicitEquations;

  llvm::DenseMap<ScheduledEquationInstanceOp, RawFunctionOp> equationFunctions;
  size_t equationFunctionsCounter = 0;

  for (SCCOp scc : SCCs) {
    if (scc.getCycle()) {
      return mlir::failure();
    }

    llvm::SmallVector<ScheduledEquationInstanceOp> equationOps;
    scc.collectEquations(equationOps);

    for (ScheduledEquationInstanceOp equationOp : equationOps) {
      auto explicitEquationOp = equationOp.cloneAndExplicitate(
          rewriter, symbolTableCollection);

      if (!explicitEquationOp) {
        return mlir::failure();
      }

      rewriter.eraseOp(equationOp);
      explicitEquations.insert(explicitEquationOp);
    }

    for (ScheduledEquationInstanceOp equationOp : explicitEquations) {
      RawFunctionOp templateFunction = createEquationTemplateFunction(
          rewriter, moduleOp, symbolTableCollection,
          equationOp.getTemplate(),
          equationOp.getViewElementIndex(),
          equationOp.getIterationDirections(),
          "equation_" + std::to_string(equationFunctionsCounter++),
          localToGlobalVariablesMap);

      if (!templateFunction) {
        return mlir::failure();
      }

      equationFunctions[equationOp] = templateFunction;
    }
  }

  if (mlir::failed(createUpdateNonStateVariablesFunction(
          rewriter, moduleOp, modelOp.getLoc(), SCCs, equationFunctions))) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateStateVariablesFunction(
          rewriter, moduleOp, modelOp.getLoc(), symbolTableCollection,
          variableOps, derivativesMap, localToGlobalVariablesMap))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult EulerForwardPass::createCalcICFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    llvm::ArrayRef<SCCOp> SCCs,
    const llvm::DenseMap<
        ScheduledEquationInstanceOp, RawFunctionOp>& equationFunctions)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "calcIC",
      builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Call the equation functions.
  for (SCCOp scc : SCCs) {
    for (ScheduledEquationInstanceOp equation :
         scc.getOps<ScheduledEquationInstanceOp>()) {
      RawFunctionOp equationRawFunction = equationFunctions.lookup(equation);

      if (mlir::failed(callEquationFunction(
              builder, loc, equation, equationRawFunction))) {
        return mlir::failure();
      }
    }
  }

  // Terminate the function.
  builder.setInsertionPointToEnd(entryBlock);
  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);

  return mlir::success();
}

mlir::LogicalResult EulerForwardPass::createUpdateNonStateVariablesFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    llvm::ArrayRef<SCCOp> SCCs,
    const llvm::DenseMap<
        ScheduledEquationInstanceOp, RawFunctionOp>& equationFunctions)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "updateNonStateVariables",
      builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Cal the equation functions.
  for (SCCOp scc : SCCs) {
    for (ScheduledEquationInstanceOp equation :
         scc.getOps<ScheduledEquationInstanceOp>()) {
      RawFunctionOp equationRawFunction = equationFunctions.lookup(equation);

      llvm::SmallVector<mlir::Value, 3> args;

      // Explicit indices.
      if (auto indices = equation.getIndices()) {
        for (size_t i = 0, e = indices->getValue().rank(); i < e; ++i) {
          // Begin index.
          args.push_back(builder.create<mlir::arith::ConstantOp>(
              loc, builder.getIndexAttr(indices->getValue()[i].getBegin())));

          // End index.
          args.push_back(builder.create<mlir::arith::ConstantOp>(
              loc, builder.getIndexAttr(indices->getValue()[i].getEnd())));

          // Step.
          args.push_back(builder.create<mlir::arith::ConstantOp>(
              loc, builder.getIndexAttr(1)));
        }
      }

      // Implicit indices.
      if (auto indices = equation.getImplicitIndices()) {
        for (size_t i = 0, e = indices->getValue().rank(); i < e; ++i) {
          // Begin index.
          args.push_back(builder.create<mlir::arith::ConstantOp>(
              loc, builder.getIndexAttr(indices->getValue()[i].getBegin())));

          // End index.
          args.push_back(builder.create<mlir::arith::ConstantOp>(
              loc, builder.getIndexAttr(indices->getValue()[i].getEnd())));

          // Step.
          args.push_back(builder.create<mlir::arith::ConstantOp>(
              loc, builder.getIndexAttr(1)));
        }
      }

      builder.create<CallOp>(loc, equationRawFunction, args);
    }
  }

  // Terminate the function.
  builder.setInsertionPointToEnd(entryBlock);
  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);

  return mlir::success();
}

mlir::LogicalResult EulerForwardPass::createUpdateStateVariablesFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    mlir::SymbolTableCollection& symbolTableCollection,
    llvm::ArrayRef<VariableOp> variableOps,
    const DerivativesMap& derivativesMap,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "updateStateVariables",
      builder.getFunctionType(builder.getF64Type(), std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  mlir::Value timeStep = functionOp.getArgument(0);

  auto apply = [&](mlir::OpBuilder& nestedBuilder,
                   mlir::Value scalarState,
                   mlir::Value scalarDerivative) -> mlir::Value {
    mlir::Value result = nestedBuilder.create<MulOp>(
        loc, scalarDerivative.getType(), scalarDerivative, timeStep);

    result = nestedBuilder.create<AddOp>(
        loc, scalarState.getType(), scalarState, result);

    return result;
  };

  for (VariableOp variableOp : variableOps) {
    if (auto derivativeName = derivativesMap.getDerivative(
            mlir::FlatSymbolRefAttr::get(variableOp.getSymNameAttr()))) {
      auto stateGlobalVarOp =
          localToGlobalVariablesMap.lookup(variableOp.getSymName());

      assert(stateGlobalVarOp && "Global variable not found");
      assert(derivativeName->getNestedReferences().empty());

      auto derivativeGlobalVarOp = localToGlobalVariablesMap.lookup(
          derivativeName->getRootReference().getValue());

      assert(derivativeGlobalVarOp && "Global derivative variable not found");
      VariableType variableType = variableOp.getVariableType();

      mlir::Value state = builder.create<GlobalVariableGetOp>(
          variableOp.getLoc(), stateGlobalVarOp);

      mlir::Value derivative = builder.create<GlobalVariableGetOp>(
          variableOp.getLoc(), derivativeGlobalVarOp);

      if (variableType.isScalar()) {
        mlir::Value stateLoad =
            builder.create<LoadOp>(state.getLoc(), state, std::nullopt);

        mlir::Value derivativeLoad = builder.create<LoadOp>(
            derivative.getLoc(), derivative, std::nullopt);

        mlir::Value updatedValue = apply(builder, stateLoad, derivativeLoad);
        builder.create<StoreOp>(loc, updatedValue, state, std::nullopt);
      } else {
        // Create the loops to iterate on each scalar variable.
        llvm::SmallVector<mlir::Value, 3> lowerBounds;
        llvm::SmallVector<mlir::Value, 3> upperBounds;
        llvm::SmallVector<mlir::Value, 3> steps;

        for (unsigned int i = 0; i < variableOp.getVariableType().getRank(); ++i) {
          lowerBounds.push_back(builder.create<ConstantOp>(
              loc, builder.getIndexAttr(0)));

          mlir::Value dimension = builder.create<ConstantOp>(
              loc, builder.getIndexAttr(i));

          upperBounds.push_back(builder.create<DimOp>(
              loc, state, dimension));

          steps.push_back(builder.create<ConstantOp>(
              loc, builder.getIndexAttr(1)));
        }

        mlir::scf::buildLoopNest(
            builder, loc, lowerBounds, upperBounds, steps,
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
  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);

  return mlir::success();
}

namespace mlir::modelica
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
