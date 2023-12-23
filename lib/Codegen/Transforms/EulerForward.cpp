#include "marco/Codegen/Transforms/EulerForward.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "marco/Codegen/Transforms/SolverPassBase.h"
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
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          llvm::ArrayRef<SCCGroupOp> sccGroups) override;

      mlir::LogicalResult solveMainModel(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variableOps,
          const DerivativesMap& derivativesMap,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          llvm::ArrayRef<SCCGroupOp> sccGroups) override;

    private:
      mlir::LogicalResult createCalcICFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          llvm::ArrayRef<SCCGroupOp> sccGroups,
          const llvm::DenseMap<
              ScheduledEquationInstanceOp, RawFunctionOp>& equationFunctions);

      mlir::LogicalResult createUpdateNonStateVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          llvm::ArrayRef<SCCGroupOp> sccGroups,
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
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    llvm::ArrayRef<SCCGroupOp> sccGroups)
{
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

  // The list of strongly connected components with cyclic dependencies.
  llvm::SmallVector<SCCOp> cycles;

  // Determine which equations can be processed by just making them explicit
  // with respect to the variable they match.
  llvm::DenseSet<ScheduledEquationInstanceOp> explicitableEquations;

  // Map between the original equations and their explicit version (if
  // computable) w.r.t. the matched variables.
  llvm::DenseMap<
      ScheduledEquationInstanceOp,
      ScheduledEquationInstanceOp> explicitEquationsMap;

  // The list of equations which could not be made explicit.
  llvm::DenseSet<ScheduledEquationInstanceOp> implicitEquations;

  // The list of equations that can be handled internally.
  llvm::DenseSet<ScheduledEquationInstanceOp> internalEquations;

  // The list of equations that are handled by external solvers.
  llvm::DenseSet<ScheduledEquationInstanceOp> externalEquations;

  auto isExternalEquationFn = [&](ScheduledEquationInstanceOp equationOp) {
    for (SCCOp scc : cycles) {
      for (ScheduledEquationInstanceOp sccEquation :
           scc.getOps<ScheduledEquationInstanceOp>()) {
        if (sccEquation == equationOp) {
          return true;
        }
      }
    }

    return implicitEquations.contains(equationOp);
  };

  // Categorize the equations.
  LLVM_DEBUG(llvm::dbgs() << "Identifying the explicitable equations\n");

  for (SCCGroupOp sccGroup : sccGroups) {
    for (SCCOp scc : sccGroup.getOps<SCCOp>()) {
      if (scc.getCycle()) {
        cycles.push_back(scc);
        continue;
      }

      // The content of an SCC may be modified, so we need to freeze the
      // initial list of equations.
      llvm::SmallVector<ScheduledEquationInstanceOp> equationOps;
      scc.collectEquations(equationOps);

      for (ScheduledEquationInstanceOp equationOp : equationOps) {
        LLVM_DEBUG({
          llvm::dbgs() << "Explicitating equation\n";
          equationOp.printInline(llvm::dbgs());
          llvm::dbgs() << "\n";
        });

        auto explicitEquationOp = equationOp.cloneAndExplicitate(
            rewriter, symbolTableCollection);

        if (explicitEquationOp) {
          LLVM_DEBUG({
            llvm::dbgs() << "Explicit equation\n";
            explicitEquationOp.printInline(llvm::dbgs());
            llvm::dbgs() << "\n";
          });

          explicitableEquations.insert(equationOp);
          explicitEquationsMap[equationOp] = explicitEquationOp;
        } else {
          LLVM_DEBUG(llvm::dbgs() << "Implicit equation found\n");
          implicitEquations.insert(equationOp);
          continue;
        }
      }
    }
  }

  // Determine the equations that can be handled internally.
  for (ScheduledEquationInstanceOp equationOp : explicitableEquations) {
    if (!isExternalEquationFn(equationOp)) {
      internalEquations.insert(equationOp);
    }
  }

  // Map from explicit equations to their algorithmically equivalent function.
  llvm::DenseMap<
      ScheduledEquationInstanceOp, RawFunctionOp> equationFunctions;

  // Create the functions for the equations managed internally.
  size_t equationFunctionsCounter = 0;

  for (ScheduledEquationInstanceOp equationOp : internalEquations) {
    ScheduledEquationInstanceOp explicitEquationOp =
        explicitEquationsMap[equationOp];

    RawFunctionOp templateFunction = createEquationTemplateFunction(
        rewriter, moduleOp, symbolTableCollection,
        explicitEquationOp.getTemplate(),
        explicitEquationOp.getViewElementIndex(),
        explicitEquationOp.getIterationDirections(),
        "initial_equation_" +
            std::to_string(equationFunctionsCounter++),
        localToGlobalVariablesMap);

    equationFunctions[equationOp] = templateFunction;
  }

  if (mlir::failed(createCalcICFunction(
          rewriter, moduleOp, modelOp.getLoc(), sccGroups, equationFunctions))) {
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
    llvm::ArrayRef<SCCGroupOp> sccGroups)
{
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();
  llvm::DenseSet<ScheduledEquationInstanceOp> explicitEquations;

  llvm::DenseMap<ScheduledEquationInstanceOp, RawFunctionOp> equationFunctions;
  size_t equationFunctionsCounter = 0;

  for (SCCGroupOp sccGroup : sccGroups) {
    for (SCCOp scc : sccGroup.getOps<SCCOp>()) {
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
  }

  if (mlir::failed(createUpdateNonStateVariablesFunction(
          rewriter, moduleOp, modelOp.getLoc(), sccGroups,
          equationFunctions))) {
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
    llvm::ArrayRef<SCCGroupOp> sccGroups,
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
  for (SCCGroupOp sccGroup : sccGroups) {
    for (SCCOp scc : sccGroup.getOps<SCCOp>()) {
      for (ScheduledEquationInstanceOp equation :
           scc.getOps<ScheduledEquationInstanceOp>()) {
        auto rawFunctionIt = equationFunctions.find(equation);

        if (rawFunctionIt == equationFunctions.end()) {
          // Equation not handled internally.
          continue;
        }

        RawFunctionOp equationRawFunction = rawFunctionIt->getSecond();

        if (mlir::failed(callEquationFunction(
                builder, loc, equation, equationRawFunction))) {
          return mlir::failure();
        }
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
    llvm::ArrayRef<SCCGroupOp> sccGroups,
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
  for (SCCGroupOp sccGroup : sccGroups) {
    for (SCCOp scc : sccGroup.getOps<SCCOp>()) {
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
