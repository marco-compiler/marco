#include "marco/Codegen/Transforms/EulerForward.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "marco/Codegen/Transforms/SolverPassBase.h"
#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "marco/Dialect/KINSOL/KINSOLDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_EULERFORWARDPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace
{
  struct ConversionInfo
  {
    std::set<std::unique_ptr<Equation>> explicitEquations;
    std::map<ScheduledEquation*, Equation*> explicitEquationsMap;
    std::set<ScheduledEquation*> implicitEquations;
    std::set<ScheduledEquation*> cyclicEquations;
  };

  /// Class to be used to uniquely identify an equation template function.
  /// Two templates are considered to be equal if they refer to the same
  /// EquationOp and have the same scheduling direction, which impacts on the
  /// function body itself due to the way the iteration indices are updated.
  class EquationTemplateInfo
  {
    public:
      EquationTemplateInfo(
          EquationInterface equation,
          scheduling::Direction schedulingDirection)
          : equation(equation.getOperation()),
            schedulingDirection(schedulingDirection)
      {
      }

      bool operator<(const EquationTemplateInfo& other) const
      {
        if (schedulingDirection != other.schedulingDirection) {
          return true;
        }

        return equation < other.equation;
      }

    private:
      mlir::Operation* equation;
      scheduling::Direction schedulingDirection;
  };

  struct EquationTemplate
  {
    mlir::func::FuncOp funcOp;
    llvm::SmallVector<VariableOp> usedVariables;
  };

  class EulerForwardSolver : public mlir::modelica::impl::ModelSolver
  {
    public:
      struct ConversionInfo
      {
        std::set<std::unique_ptr<Equation>> explicitEquations;
        std::map<ScheduledEquation*, Equation*> explicitEquationsMap;
        std::set<ScheduledEquation*> implicitEquations;
        std::set<ScheduledEquation*> cyclicEquations;
      };

    protected:
      mlir::LogicalResult solveICModel(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          const marco::codegen::Model<
              marco::codegen::ScheduledEquationsBlock>& model) override;

      mlir::LogicalResult solveMainModel(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          const marco::codegen::Model<
              marco::codegen::ScheduledEquationsBlock>& model) override;

    private:
      /// Create the function that computes the initial conditions.
      mlir::LogicalResult createCalcICFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          const Model<ScheduledEquationsBlock>& model,
          const ConversionInfo& conversionInfo) const;

      /// Create the functions that calculates the values that the non-state
      /// variables will have in the next iteration.
      mlir::LogicalResult createUpdateNonStateVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          const Model<ScheduledEquationsBlock>& model,
          const ConversionInfo& conversionInfo) const;

      /// Create the functions that calculates the values that the state
      /// variables will have in the next iteration.
      mlir::LogicalResult createUpdateStateVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          const Model<ScheduledEquationsBlock>& model) const;

      mlir::func::FuncOp createEquationFunction(
          mlir::OpBuilder& builder,
          const ScheduledEquation& equation,
          llvm::StringRef equationFunctionName,
          mlir::func::FuncOp templateFunction,
        mlir::TypeRange varsTypes) const;
  };
}

/// Get or create the template equation function for a scheduled equation.
static llvm::Optional<EquationTemplate> getOrCreateEquationTemplateFunction(
    llvm::ThreadPool& threadPool,
    mlir::OpBuilder& builder,
    ModelOp modelOp,
    const mlir::SymbolTable& symbolTable,
    const ScheduledEquation* equation,
    const EulerForwardSolver::ConversionInfo& conversionInfo,
    std::map<EquationTemplateInfo, EquationTemplate>& equationTemplatesMap,
    llvm::StringRef functionNamePrefix,
    size_t& equationTemplateCounter)
{
  EquationInterface equationInt = equation->getOperation();
  EquationTemplateInfo requestedTemplate(equationInt, equation->getSchedulingDirection());

  // Check if the template equation already exists.
  if (auto it = equationTemplatesMap.find(requestedTemplate); it != equationTemplatesMap.end()) {
    return it->second;
  }

  // If not, create it.
  mlir::Location loc = equationInt.getLoc();

  // Name of the function.
  std::string functionName = functionNamePrefix.str() + std::to_string(equationTemplateCounter++);

  auto explicitEquation = llvm::find_if(conversionInfo.explicitEquationsMap, [&](const auto& equationPtr) {
    return equationPtr.first == equation;
  });

  if (explicitEquation == conversionInfo.explicitEquationsMap.end()) {
    // The equation can't be made explicit and is not passed to any external solver
    return llvm::None;
  }

  // Create the equation template function
  llvm::SmallVector<VariableOp> usedVariables;

  mlir::func::FuncOp function = explicitEquation->second->createTemplateFunction(
      threadPool, builder, functionName,
      equation->getSchedulingDirection(),
      symbolTable,
      usedVariables);

  size_t timeArgumentIndex = 0;

  function.insertArgument(
      timeArgumentIndex,
      RealType::get(builder.getContext()),
      builder.getDictionaryAttr(llvm::None),
      loc);

  function.walk([&](TimeOp timeOp) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(timeOp);
    timeOp.replaceAllUsesWith(function.getArgument(timeArgumentIndex));
    timeOp.erase();
  });

  EquationTemplate result{function, usedVariables};
  equationTemplatesMap[requestedTemplate] = result;
  return result;
}

mlir::LogicalResult EulerForwardSolver::solveICModel(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    const marco::codegen::Model<
        marco::codegen::ScheduledEquationsBlock>& model)
{
  ConversionInfo conversionInfo;

  // Determine which equations can be processed, that is:
  //  - Those that can me bade explicit with respect to their matched
  //    variable.
  //  - Those that do not take part to a cycle.

  for (auto& scheduledBlock : model.getScheduledBlocks()) {
    if (!scheduledBlock->hasCycle()) {
      for (auto& scheduledEquation : *scheduledBlock) {
        auto explicitClone = scheduledEquation->cloneIRAndExplicitate(builder);

        if (explicitClone == nullptr) {
          conversionInfo.implicitEquations.emplace(scheduledEquation.get());
        } else {
          auto& movedClone = *conversionInfo.explicitEquations.emplace(std::move(explicitClone)).first;
          conversionInfo.explicitEquationsMap[scheduledEquation.get()] = movedClone.get();
        }
      }
    } else {
      for (const auto& equation : *scheduledBlock) {
        conversionInfo.cyclicEquations.emplace(equation.get());
      }
    }
  }

  // Fail in case of implicit equations.
  if (!conversionInfo.implicitEquations.empty()) {
    return mlir::failure();
  }

  // Fail in case of cycles.
  if (!conversionInfo.cyclicEquations.empty()) {
    return mlir::failure();
  }

  if (mlir::failed(createCalcICFunction(
          builder, simulationModuleOp, model, conversionInfo))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult EulerForwardSolver::solveMainModel(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    const marco::codegen::Model<
        marco::codegen::ScheduledEquationsBlock>& model)
{
  ConversionInfo conversionInfo;

  // Determine which equations can be processed, that is:
  //  - Those that can me bade explicit with respect to their matched
  //    variable.
  //  - Those that do not take part to a cycle.

  for (auto& scheduledBlock : model.getScheduledBlocks()) {
    if (!scheduledBlock->hasCycle()) {
      for (auto& scheduledEquation : *scheduledBlock) {
        auto explicitClone = scheduledEquation->cloneIRAndExplicitate(builder);

        if (explicitClone == nullptr) {
          conversionInfo.implicitEquations.emplace(scheduledEquation.get());
        } else {
          auto& movedClone = *conversionInfo.explicitEquations.emplace(std::move(explicitClone)).first;
          conversionInfo.explicitEquationsMap[scheduledEquation.get()] = movedClone.get();
        }
      }
    } else {
      for (const auto& equation : *scheduledBlock) {
        conversionInfo.cyclicEquations.emplace(equation.get());
      }
    }
  }

  // Fail in case of implicit equations.
  if (!conversionInfo.implicitEquations.empty()) {
    return mlir::failure();
  }

  // Fail in case of cycles.
  if (!conversionInfo.cyclicEquations.empty()) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateNonStateVariablesFunction(
          builder, simulationModuleOp, model, conversionInfo))) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateStateVariablesFunction(
          builder, simulationModuleOp, model))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult EulerForwardSolver::createUpdateNonStateVariablesFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    const Model<ScheduledEquationsBlock>& model,
    const ConversionInfo& conversionInfo) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto modelOp = model.getOperation();
  mlir::SymbolTable symbolTable(modelOp);
  auto loc = modelOp.getLoc();

  // Create the function inside the parent module.
  builder.setInsertionPointToEnd(simulationModuleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "updateNonStateVariables",
      llvm::None,
      RealType::get(builder.getContext()),
      simulationModuleOp.getVariablesTypes(),
      llvm::None, llvm::None);

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  llvm::SmallVector<mlir::Value> allVariables;
  allVariables.push_back(functionOp.getTime());

  for (mlir::Value variable : functionOp.getVariables()) {
    allVariables.push_back(variable);
  }

  // Convert the equations into algorithmic code.
  size_t equationTemplateCounter = 0;
  size_t equationCounter = 0;

  std::map<EquationTemplateInfo, EquationTemplate> equationTemplatesMap;
  llvm::ThreadPool threadPool;

  llvm::StringMap<size_t> variablesPos;

  for (auto variable : llvm::enumerate(modelOp.getVariables())) {
    variablesPos[variable.value().getSymName()] = variable.index();
  }

  for (const auto& scheduledBlock : model.getScheduledBlocks()) {
    for (const auto& equation : *scheduledBlock) {
      auto templateFunction = getOrCreateEquationTemplateFunction(
          threadPool, builder, modelOp, symbolTable, equation.get(), conversionInfo,
          equationTemplatesMap, "eq_template_", equationTemplateCounter);

      if (!templateFunction.has_value()) {
        equation->getOperation().emitError(
            "The equation can't be made explicit");

        equation->getOperation().dump();
        return mlir::failure();
      }

      // Create the function that calls the template.
      // This function dictates the indices the template will work with.
      std::string equationFunctionName =
          "eq_" + std::to_string(equationCounter);

      ++equationCounter;

      // Collect the variables to be passed to the instantiated template
      // function.
      std::vector<mlir::Value> usedVariables;
      std::vector<mlir::Type> usedVariablesTypes;

      usedVariables.push_back(allVariables[0]);
      usedVariablesTypes.push_back(allVariables[0].getType());

      for (VariableOp variableOp : templateFunction->usedVariables) {
        usedVariables.push_back(allVariables[variablesPos[variableOp.getSymName()] + 1]);
        usedVariablesTypes.push_back(allVariables[variablesPos[variableOp.getSymName()] + 1].getType());
      }

      auto equationFunction = createEquationFunction(
          builder, *equation, equationFunctionName,
          templateFunction->funcOp, usedVariablesTypes);

      // Create the call to the instantiated template function
      builder.create<mlir::func::CallOp>(loc, equationFunction, usedVariables);
    }
  }

  // Terminate the function.
  builder.create<mlir::simulation::ReturnOp>(loc, llvm::None);

  return mlir::success();
}

mlir::LogicalResult EulerForwardSolver::createCalcICFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    const Model<ScheduledEquationsBlock>& model,
    const ConversionInfo& conversionInfo) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto modelOp = model.getOperation();
  mlir::SymbolTable symbolTable(modelOp);
  auto loc = modelOp.getLoc();

  // Create the function inside the parent module.
  builder.setInsertionPointToEnd(simulationModuleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "calcIC",
      llvm::None,
      RealType::get(builder.getContext()),
      simulationModuleOp.getVariablesTypes(),
      llvm::None, llvm::None);

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  llvm::SmallVector<mlir::Value> allVariables;
  allVariables.push_back(functionOp.getTime());

  for (mlir::Value variable : functionOp.getVariables()) {
    allVariables.push_back(variable);
  }

  // Convert the equations into algorithmic code.
  size_t equationTemplateCounter = 0;
  size_t equationCounter = 0;

  std::map<EquationTemplateInfo, EquationTemplate> equationTemplatesMap;
  llvm::ThreadPool threadPool;

  llvm::StringMap<size_t> variablesPos;

  for (auto variable : llvm::enumerate(modelOp.getVariables())) {
    variablesPos[variable.value().getSymName()] = variable.index();
  }

  for (const auto& scheduledBlock : model.getScheduledBlocks()) {
    for (const auto& equation : *scheduledBlock) {
      auto templateFunction = getOrCreateEquationTemplateFunction(
          threadPool, builder, modelOp, symbolTable, equation.get(), conversionInfo,
          equationTemplatesMap, "initial_eq_template_", equationTemplateCounter);

      if (!templateFunction.has_value()) {
        equation->getOperation().emitError("The equation can't be made explicit");
        equation->getOperation().dump();
        return mlir::failure();
      }

      // Create the function that calls the template.
      // This function dictates the indices the template will work with.
      std::string equationFunctionName = "initial_eq_" + std::to_string(equationCounter);
      ++equationCounter;

      // Collect the variables to be passed to the instantiated template function.
      std::vector<mlir::Value> usedVariables;
      std::vector<mlir::Type> usedVariablesTypes;

      usedVariables.push_back(allVariables[0]);
      usedVariablesTypes.push_back(allVariables[0].getType());

      for (VariableOp variableOp : templateFunction->usedVariables) {
        usedVariables.push_back(allVariables[variablesPos[variableOp.getSymName()] + 1]);
        usedVariablesTypes.push_back(allVariables[variablesPos[variableOp.getSymName()] + 1].getType());
      }

      // Create the instantiated template function.
      auto equationFunction = createEquationFunction(
          builder, *equation, equationFunctionName,
          templateFunction->funcOp, usedVariablesTypes);

      // Create the call to the instantiated template function.
      builder.create<mlir::func::CallOp>(loc, equationFunction, usedVariables);
    }
  }

  // Terminate the function.
  builder.create<mlir::simulation::ReturnOp>(loc, llvm::None);

  return mlir::success();
}

mlir::LogicalResult EulerForwardSolver::createUpdateStateVariablesFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    const Model<ScheduledEquationsBlock>& model) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto modelOp = model.getOperation();
  mlir::Location loc = modelOp.getLoc();

  // Create the function inside the parent module.
  builder.setInsertionPointToEnd(simulationModuleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "updateStateVariables",
      llvm::None,
      RealType::get(builder.getContext()),
      simulationModuleOp.getVariablesTypes(),
      builder.getF64Type(),
      llvm::None);

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  llvm::SmallVector<mlir::Value> allVariables;
  allVariables.push_back(functionOp.getTime());

  for (mlir::Value variable : functionOp.getVariables()) {
    allVariables.push_back(variable);
  }

  // Update the state variables by applying the forward Euler method.
  mlir::Value timeStep = functionOp.getExtraArgs()[0];

  auto apply = [&](mlir::OpBuilder& nestedBuilder,
                   mlir::Value scalarState,
                   mlir::Value scalarDerivative) -> mlir::Value {
    mlir::Value result = nestedBuilder.create<MulOp>(
        loc, scalarDerivative.getType(), scalarDerivative, timeStep);

    result = nestedBuilder.create<AddOp>(
        loc, scalarState.getType(), scalarState, result);

    return result;
  };

  auto variables = functionOp.getVariables();
  const DerivativesMap& derivativesMap = model.getDerivativesMap();

  llvm::StringMap<size_t> variablesPos;

  for (const auto& variableAttr :
       llvm::enumerate(simulationModuleOp.getVariables()
           .getAsRange<mlir::simulation::VariableAttr>())) {
    llvm::StringRef variableName = variableAttr.value().getName();
    variablesPos[variableName] = variableAttr.index();
  }

  for (const auto& var : model.getVariables()) {
    VariableOp variableOp = var->getDefiningOp();
    llvm::StringRef variableName = variableOp.getSymName();
    mlir::Value variable = variables[variablesPos[variableName]];

    if (derivativesMap.hasDerivative(variableName)) {
      auto derVarName = derivativesMap.getDerivative(variableName);
      mlir::Value derivative = variables[variablesPos[derVarName]];

      if (variableOp.getVariableType().isScalar()) {
        mlir::Value scalarState = builder.create<LoadOp>(
            loc, variable, llvm::None);

        mlir::Value scalarDerivative = builder.create<LoadOp>(
            loc, derivative, llvm::None);

        mlir::Value updatedValue = apply(
            builder, scalarState, scalarDerivative);

        builder.create<StoreOp>(loc, updatedValue, variable, llvm::None);

      } else {
        // Create the loops to iterate on each scalar variable.
        std::vector<mlir::Value> lowerBounds;
        std::vector<mlir::Value> upperBounds;
        std::vector<mlir::Value> steps;

        for (unsigned int i = 0; i < variableOp.getVariableType().getRank(); ++i) {
          lowerBounds.push_back(builder.create<ConstantOp>(
              loc, builder.getIndexAttr(0)));

          mlir::Value dimension = builder.create<ConstantOp>(
              loc, builder.getIndexAttr(i));

          upperBounds.push_back(builder.create<DimOp>(
              loc, variable, dimension));

          steps.push_back(builder.create<ConstantOp>(
              loc, builder.getIndexAttr(1)));
        }

        mlir::scf::buildLoopNest(
            builder, loc, lowerBounds, upperBounds, steps,
            [&](mlir::OpBuilder& nestedBuilder,
                mlir::Location loc,
                mlir::ValueRange indices) {
              mlir::Value scalarState = nestedBuilder.create<LoadOp>(
                  loc, variable, indices);

              mlir::Value scalarDerivative = nestedBuilder.create<LoadOp>(
                  loc, derivative, indices);

              mlir::Value updatedValue = apply(
                  nestedBuilder, scalarState, scalarDerivative);

              nestedBuilder.create<StoreOp>(
                  loc, updatedValue, variable, indices);
            });
      }
    }
  }

  // Terminate the function.
  builder.create<mlir::simulation::ReturnOp>(loc, llvm::None);

  return mlir::success();
}

mlir::func::FuncOp EulerForwardSolver::createEquationFunction(
    mlir::OpBuilder& builder,
    const ScheduledEquation& equation,
    llvm::StringRef equationFunctionName,
    mlir::func::FuncOp templateFunction,
    mlir::TypeRange varsTypes) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = equation.getOperation().getLoc();

  auto module = equation.getOperation()->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(module.getBody());

  // Function type. The equation doesn't need to return any value.
  // Initially we consider all the variables as to be passed to the function.
  // We will then remove the unused ones.
  auto functionType = builder.getFunctionType(varsTypes, llvm::None);

  auto function = builder.create<mlir::func::FuncOp>(loc, equationFunctionName, functionType);

  function->setAttr(
      "llvm.linkage",
      mlir::LLVM::LinkageAttr::get(
          builder.getContext(), mlir::LLVM::Linkage::Internal));

  auto* entryBlock = function.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto valuesFn = [&](marco::modeling::scheduling::Direction iterationDirection, Range range) -> std::tuple<mlir::Value, mlir::Value, mlir::Value> {
    assert(iterationDirection == marco::modeling::scheduling::Direction::Forward ||
           iterationDirection == marco::modeling::scheduling::Direction::Backward);

    if (iterationDirection == marco::modeling::scheduling::Direction::Forward) {
      mlir::Value begin = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(range.getBegin()));
      mlir::Value end = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(range.getEnd()));
      mlir::Value step = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(1));

      return std::make_tuple(begin, end, step);
    }

    mlir::Value begin = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(range.getEnd() - 1));
    mlir::Value end = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(range.getBegin() - 1));
    mlir::Value step = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(1));

    return std::make_tuple(begin, end, step);
  };

  mlir::ValueRange vars = function.getArguments();
  std::vector<mlir::Value> args(vars.begin(), vars.end());

  auto iterationRangesSet = equation.getIterationRanges();
  assert(iterationRangesSet.isSingleMultidimensionalRange());//todo: handle ragged case
  auto iterationRanges = iterationRangesSet.minContainingRange();

  for (size_t i = 0, e = equation.getNumOfIterationVars(); i < e; ++i) {
    auto values = valuesFn(equation.getSchedulingDirection(), iterationRanges[i]);

    args.push_back(std::get<0>(values));
    args.push_back(std::get<1>(values));
    args.push_back(std::get<2>(values));
  }

  // Call the equation template function
  builder.create<mlir::func::CallOp>(loc, templateFunction, args);

  builder.create<mlir::func::ReturnOp>(loc);
  return function;
}

namespace
{
  class EulerForwardPass
      : public mlir::modelica::impl::EulerForwardPassBase<EulerForwardPass>
  {
    public:
      using EulerForwardPassBase::EulerForwardPassBase;

      void runOnOperation() override
      {
        mlir::ModuleOp module = getOperation();
        std::vector<ModelOp> modelOps;

        module.walk([&](ModelOp modelOp) {
          modelOps.push_back(modelOp);
        });

        assert(llvm::count_if(modelOps, [&](ModelOp modelOp) {
                 return modelOp.getSymName() == model;
               }) <= 1 && "More than one model matches the requested model name, but only one can be converted into a simulation");

        EulerForwardSolver solver;

        auto expectedVariablesFilter = marco::VariableFilter::fromString(variablesFilter);
        std::unique_ptr<marco::VariableFilter> variablesFilterInstance;

        if (!expectedVariablesFilter) {
          getOperation().emitWarning("Invalid variable filter string. No filtering will take place");
          variablesFilterInstance = std::make_unique<marco::VariableFilter>();
        } else {
          variablesFilterInstance = std::make_unique<marco::VariableFilter>(std::move(*expectedVariablesFilter));
        }

        for (ModelOp modelOp : modelOps) {
          if (mlir::failed(solver.convert(
                  modelOp, *variablesFilterInstance,
                  processICModel, processMainModel))) {
            return signalPassFailure();
          }
        }

        mlir::modelica::TypeConverter typeConverter(bitWidth);

        if (mlir::failed(solver.legalizeFuncOps(
                getOperation(), typeConverter))) {
          return signalPassFailure();
        }
      }
  };
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
