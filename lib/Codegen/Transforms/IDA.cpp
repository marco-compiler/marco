#include "marco/Codegen/Transforms/IDA.h"
#include "marco/Codegen/Transforms/SolverPassBase.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Dialect/KINSOL/KINSOLDialect.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "marco/Codegen/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_IDAPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace
{
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

    // Positions of the used variables.
    std::vector<unsigned int> usedVariables;
  };
}

namespace
{
  class IDAInstance
  {
    public:
      IDAInstance(const DerivativesMap& derivativesMap, bool cleverDAE);

      void setStartTime(double time);

      void setEndTime(double time);

      bool hasVariable(mlir::Value variable) const;

      void addParametricVariable(mlir::Value variable);

      void addStateVariable(mlir::Value variable);

      void addDerivativeVariable(mlir::Value variable);

      void addAlgebraicVariable(mlir::Value variable);

      bool hasEquation(ScheduledEquation* equation) const;

      void addEquation(ScheduledEquation* equation);

      mlir::Value createInstance(
          mlir::OpBuilder& builder,
          mlir::Location loc);

      mlir::LogicalResult configure(
          mlir::OpBuilder& builder,
          mlir::Value idaInstance,
          const Model<ScheduledEquationsBlock>& model,
          mlir::ValueRange variables);

      mlir::LogicalResult performCalcIC(
          mlir::OpBuilder& builder,
          mlir::Value idaInstance);

      mlir::LogicalResult performStep(
          mlir::OpBuilder& builder,
          mlir::Value idaInstance);

      mlir::Value getCurrentTime(
          mlir::OpBuilder& builder,
          mlir::Value solverDataPtr,
          mlir::Type timeType);

      mlir::LogicalResult deleteInstance(
          mlir::OpBuilder& builder,
          mlir::Value instance);

    private:
      bool hasParametricVariable(mlir::Value variable) const;

      bool hasAlgebraicVariable(mlir::Value variable) const;

      bool hasStateVariable(mlir::Value variable) const;

      bool hasDerivativeVariable(mlir::Value variable) const;

      mlir::LogicalResult addVariablesToIDA(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Value idaInstance,
          mlir::ValueRange variables);

      mlir::ida::VariableGetterOp createGetterFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Location loc,
          mlir::Type variableType,
          llvm::StringRef functionName);

      mlir::ida::VariableSetterOp createSetterFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Location loc,
          mlir::Type variableType,
          llvm::StringRef functionName);

      mlir::LogicalResult addEquationsToIDA(
          mlir::OpBuilder& builder,
          mlir::Value idaInstance,
          const Model<ScheduledEquationsBlock>& model);

      mlir::LogicalResult addVariableAccessesInfoToIDA(
          mlir::OpBuilder& builder,
          mlir::Value idaInstance,
          const Equation& equation,
          mlir::Value idaEquation);

      mlir::LogicalResult createResidualFunction(
          mlir::OpBuilder& builder,
          const Equation& equation,
          mlir::Value idaEquation,
          llvm::StringRef residualFunctionName);

      mlir::LogicalResult createPartialDerTemplateFunction(
          mlir::OpBuilder& builder,
          const Equation& equation,
          llvm::StringRef templateName);

      mlir::modelica::FunctionOp createPartialDerTemplateFromEquation(
          mlir::OpBuilder& builder,
          const marco::codegen::Equation& equation,
          llvm::StringRef templateName);

      mlir::LogicalResult createJacobianFunction(
          mlir::OpBuilder& builder,
          const Equation& equation,
          llvm::StringRef jacobianFunctionName,
          mlir::Value independentVariable,
          llvm::StringRef partialDerTemplateName);

      std::vector<mlir::Value> getIDAFunctionArgs() const;

      std::multimap<
          unsigned int,
          std::pair<IndexSet, ScheduledEquation*>>
      getWritesMap(const Model<ScheduledEquationsBlock>& model) const;

      mlir::AffineMap getAccessMap(
          mlir::OpBuilder& builder,
          const AccessFunction& accessFunction) const;

    private:
      const DerivativesMap* derivativesMap;

      bool cleverDAE;

      llvm::Optional<double> startTime;
      llvm::Optional<double> endTime;

      /// The parametric variables of the model that are managed by IDA.
      /// A parametric variable is a variable that is immutable.
      /// The SSA values are the ones defined by the body of the ModelOp.
      llvm::SmallVector<mlir::Value> parametricVariables;

      /// The algebraic variables of the model that are managed by IDA.
      /// An algebraic variable is a variable that is not a parameter, state or
      /// derivative.
      /// The SSA values are the ones defined by the body of the ModelOp.
      llvm::SmallVector<mlir::Value> algebraicVariables;

      /// The state variables of the model that are managed by IDA.
      /// A state variable is a variable for which there exists a derivative
      /// variable.
      /// The SSA values are the ones defined by the body of the ModelOp.
      llvm::SmallVector<mlir::Value> stateVariables;

      /// The derivative variables of the model that are managed by IDA.
      /// A derivative variable is a variable that is the derivative of another
      /// variable.
      /// The SSA values are the ones defined by the body of the ModelOp.
      llvm::SmallVector<mlir::Value> derivativeVariables;

      /// The SSA values of the IDA variables representing the algebraic ones.
      llvm::SmallVector<mlir::Value> idaAlgebraicVariables;

      /// The SSA values of the IDA variables representing the state ones.
      llvm::SmallVector<mlir::Value> idaStateVariables;

      /// Map used for a faster lookup of the parametric variable position
      /// given its number.
      llvm::DenseMap<unsigned int, size_t> parametricVariablesLookup;

      /// Map used for a faster lookup of the algebraic variable position given
      /// its number.
      llvm::DenseMap<unsigned int, size_t> algebraicVariablesLookup;

      /// Map used for a faster lookup of the state variable position given
      /// its number.
      llvm::DenseMap<unsigned int, size_t> stateVariablesLookup;

      /// Map used for a faster lookup of the derivative variable position
      /// given its number.
      llvm::DenseMap<unsigned int, size_t> derivativeVariablesLookup;

      /// The equations managed by IDA.
      llvm::DenseSet<ScheduledEquation*> equations;
  };
}

IDAInstance::IDAInstance(
    const DerivativesMap& derivativesMap,
    bool cleverDAE)
    : derivativesMap(&derivativesMap),
      cleverDAE(cleverDAE),
      startTime(llvm::None),
      endTime(llvm::None)
{
}

void IDAInstance::setStartTime(double time)
{
  startTime = time;
}

void IDAInstance::setEndTime(double time)
{
  endTime = time;
}

bool IDAInstance::hasVariable(mlir::Value variable) const
{
  return hasParametricVariable(variable) ||
      hasAlgebraicVariable(variable) ||
      hasStateVariable(variable) ||
      hasDerivativeVariable(variable);
}

void IDAInstance::addParametricVariable(mlir::Value variable)
{
  if (!hasVariable(variable)) {
    unsigned int argNumber =
        variable.cast<mlir::BlockArgument>().getArgNumber();

    parametricVariables.push_back(variable);
    parametricVariablesLookup[argNumber] = parametricVariables.size() - 1;
  }
}

void IDAInstance::addAlgebraicVariable(mlir::Value variable)
{
  if (!hasVariable(variable)) {
    unsigned int argNumber =
        variable.cast<mlir::BlockArgument>().getArgNumber();

    algebraicVariables.push_back(variable);
    algebraicVariablesLookup[argNumber] = algebraicVariables.size() - 1;
  }
}

void IDAInstance::addStateVariable(mlir::Value variable)
{
  if (!hasVariable(variable)) {
    unsigned int argNumber =
        variable.cast<mlir::BlockArgument>().getArgNumber();

    stateVariables.push_back(variable);
    stateVariablesLookup[argNumber] = stateVariables.size() - 1;
  }
}

void IDAInstance::addDerivativeVariable(mlir::Value variable)
{
  if (!hasVariable(variable)) {
    unsigned int argNumber =
        variable.cast<mlir::BlockArgument>().getArgNumber();

    derivativeVariables.push_back(variable);
    derivativeVariablesLookup[argNumber] = derivativeVariables.size() - 1;
  }
}

bool IDAInstance::hasParametricVariable(mlir::Value variable) const
{
  unsigned int argNumber =
      variable.cast<mlir::BlockArgument>().getArgNumber();

  return parametricVariablesLookup.find(argNumber) !=
      parametricVariablesLookup.end();
}

bool IDAInstance::hasAlgebraicVariable(mlir::Value variable) const
{
  unsigned int argNumber =
      variable.cast<mlir::BlockArgument>().getArgNumber();

  return algebraicVariablesLookup.find(argNumber) !=
      algebraicVariablesLookup.end();
}

bool IDAInstance::hasStateVariable(mlir::Value variable) const
{
  unsigned int argNumber =
      variable.cast<mlir::BlockArgument>().getArgNumber();

  return stateVariablesLookup.find(argNumber) != stateVariablesLookup.end();
}

bool IDAInstance::hasDerivativeVariable(mlir::Value variable) const
{
  unsigned int argNumber =
      variable.cast<mlir::BlockArgument>().getArgNumber();

  return derivativeVariablesLookup.find(argNumber) !=
      derivativeVariablesLookup.end();
}

bool IDAInstance::hasEquation(ScheduledEquation* equation) const
{
  return llvm::find(equations, equation) != equations.end();
}

void IDAInstance::addEquation(ScheduledEquation* equation)
{
  equations.insert(equation);
}

mlir::Value IDAInstance::createInstance(
    mlir::OpBuilder& builder,
    mlir::Location loc)
{
  // Create the IDA instance.
  // To create the IDA instance, we need to first compute the total number of
  // scalar variables that IDA has to manage. Such number is equal to the
  // number of scalar equations.

  size_t numberOfScalarEquations = 0;

  for (const auto& equation : equations) {
    numberOfScalarEquations += equation->getIterationRanges().flatSize();
  }

  return builder.create<mlir::ida::CreateOp>(
      loc, builder.getI64IntegerAttr(numberOfScalarEquations));
}

mlir::LogicalResult IDAInstance::deleteInstance(
    mlir::OpBuilder& builder,
    mlir::Value instance)
{
  builder.create<mlir::ida::FreeOp>(instance.getLoc(), instance);
  return mlir::success();
}

mlir::LogicalResult IDAInstance::configure(
    mlir::OpBuilder& builder,
    mlir::Value idaInstance,
    const Model<ScheduledEquationsBlock>& model,
    mlir::ValueRange variables)
{
  auto moduleOp = model.getOperation()->getParentOfType<mlir::ModuleOp>();

  if (startTime.has_value()) {
    builder.create<mlir::ida::SetStartTimeOp>(
        idaInstance.getLoc(), idaInstance,
        builder.getF64FloatAttr(*startTime));
  }

  if (endTime.has_value()) {
    builder.create<mlir::ida::SetEndTimeOp>(
        idaInstance.getLoc(), idaInstance,
        builder.getF64FloatAttr(*endTime));
  }

  // Add the variables to IDA.
  if (mlir::failed(addVariablesToIDA(
          builder, moduleOp, idaInstance, variables))) {
    return mlir::failure();
  }

  // Add the equations to IDA.
  if (mlir::failed(addEquationsToIDA(builder, idaInstance, model))) {
    return mlir::failure();
  }

  // Initialize the IDA instance.
  builder.create<mlir::ida::InitOp>(idaInstance.getLoc(), idaInstance);

  return mlir::success();
}

mlir::LogicalResult IDAInstance::addVariablesToIDA(
    mlir::OpBuilder& builder,
    mlir::ModuleOp module,
    mlir::Value idaInstance,
    mlir::ValueRange variables)
{
  mlir::Location loc = idaInstance.getLoc();

  // Counters used to generate unique names for the getter and setter
  // functions.
  unsigned int getterFunctionCounter = 0;
  unsigned int setterFunctionCounter = 0;

  // Function to get the dimensions of a variable.
  auto getDimensionsFn = [](ArrayType arrayType) -> std::vector<int64_t> {
    assert(arrayType.hasStaticShape());

    std::vector<int64_t> dimensions;

    if (arrayType.isScalar()) {
      // In case of scalar variables, the shape of the array would be empty
      // but IDA needs to see a single dimension of value 1.
      dimensions.push_back(1);
    } else {
      auto shape = arrayType.getShape();
      dimensions.insert(dimensions.end(), shape.begin(), shape.end());
    }

    return dimensions;
  };

  auto getOrCreateGetterFn = [&](ArrayType arrayType) {
    std::string getterName = getUniqueSymbolName(module, [&]() {
      return "ida_getter_" + std::to_string(getterFunctionCounter++);
    });

    return createGetterFunction(builder, module, loc, arrayType, getterName);
  };

  auto getOrCreateSetterFn = [&](ArrayType arrayType) {
    std::string setterName = getUniqueSymbolName(module, [&]() {
      return "ida_setter_" + std::to_string(setterFunctionCounter++);
    });

    return createSetterFunction(builder, module, loc, arrayType, setterName);
  };

  // Parametric variables.
  for (mlir::Value variable : parametricVariables) {
    unsigned int argNumber =
        variable.cast<mlir::BlockArgument>().getArgNumber();

    builder.create<mlir::ida::AddParametricVariableOp>(
        loc, idaInstance, variables[argNumber]);
  }

  // Algebraic variables.
  for (mlir::Value variable : algebraicVariables) {
    unsigned int argNumber =
        variable.cast<mlir::BlockArgument>().getArgNumber();

    auto arrayType = variable.getType().cast<ArrayType>();

    std::vector<int64_t> dimensions = getDimensionsFn(arrayType);
    auto getter = getOrCreateGetterFn(arrayType);
    auto setter = getOrCreateSetterFn(arrayType);

    mlir::Value idaVariable =
        builder.create<mlir::ida::AddAlgebraicVariableOp>(
            loc,
            idaInstance,
            variables[argNumber],
            builder.getI64ArrayAttr(dimensions),
            getter.getSymName(),
            setter.getSymName());

    idaAlgebraicVariables.push_back(idaVariable);
  }

  // State variables.
  for (mlir::Value variable : stateVariables) {
    unsigned int argNumber =
        variable.cast<mlir::BlockArgument>().getArgNumber();

    auto arrayType = variable.getType().cast<ArrayType>();

    std::vector<int64_t> dimensions = getDimensionsFn(arrayType);
    auto getter = getOrCreateGetterFn(arrayType);
    auto setter = getOrCreateSetterFn(arrayType);

    mlir::Value idaVariable =
        builder.create<mlir::ida::AddStateVariableOp>(
            loc,
            idaInstance,
            variables[argNumber],
            builder.getI64ArrayAttr(dimensions),
            getter.getSymName(),
            setter.getSymName());

    idaStateVariables.push_back(idaVariable);
  }

  // Derivative variables.
  for (auto stateVariable : llvm::enumerate(stateVariables)) {
    unsigned stateArgNumber =
        stateVariable.value().cast<mlir::BlockArgument>().getArgNumber();

    unsigned int derivativeArgNumber =
        derivativesMap->getDerivative(stateArgNumber);

    mlir::Value derivativeVariable = variables[derivativeArgNumber];

    auto arrayType = derivativeVariable.getType().cast<ArrayType>();

    std::vector<int64_t> dimensions = getDimensionsFn(arrayType);
    auto getter = getOrCreateGetterFn(arrayType);
    auto setter = getOrCreateSetterFn(arrayType);

    builder.create<mlir::ida::SetDerivativeOp>(
        loc,
        idaInstance,
        idaStateVariables[stateVariable.index()],
        derivativeVariable,
        getter.getSymName(),
        setter.getSymName());
  }

  return mlir::success();
}

mlir::ida::VariableGetterOp IDAInstance::createGetterFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp module,
    mlir::Location loc,
    mlir::Type variableType,
    llvm::StringRef functionName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());

  assert(variableType.isa<ArrayType>());
  auto variableArrayType = variableType.cast<ArrayType>();

  auto getterOp = builder.create<mlir::ida::VariableGetterOp>(
      loc,
      functionName,
      RealType::get(builder.getContext()),
      variableArrayType,
      std::max(static_cast<int64_t>(1), variableArrayType.getRank()));

  mlir::Block* entryBlock = getterOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto receivedIndices = getterOp.getVariableIndices().take_front(variableArrayType.getRank());
  mlir::Value result = builder.create<LoadOp>(loc, getterOp.getVariable(), receivedIndices);

  if (auto requestedResultType = getterOp.getFunctionType().getResult(0); result.getType() != requestedResultType) {
    result = builder.create<CastOp>(loc, requestedResultType, result);
  }

  builder.create<mlir::ida::ReturnOp>(loc, result);

  return getterOp;
}

mlir::ida::VariableSetterOp IDAInstance::createSetterFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp module,
    mlir::Location loc,
    mlir::Type variableType,
    llvm::StringRef functionName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());

  assert(variableType.isa<ArrayType>());
  auto variableArrayType = variableType.cast<ArrayType>();

  auto setterOp = builder.create<mlir::ida::VariableSetterOp>(
      loc,
      functionName,
      variableArrayType,
      variableArrayType.getElementType(),
      std::max(static_cast<int64_t>(1), variableArrayType.getRank()));

  mlir::Block* entryBlock = setterOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto receivedIndices = setterOp.getVariableIndices().take_front(variableArrayType.getRank());
  mlir::Value value = setterOp.getValue();

  if (auto requestedValueType = variableArrayType.getElementType(); value.getType() != requestedValueType) {
    value = builder.create<CastOp>(loc, requestedValueType, setterOp.getValue());
  }

  builder.create<StoreOp>(loc, value, setterOp.getVariable(), receivedIndices);
  builder.create<mlir::ida::ReturnOp>(loc);

  return setterOp;
}

mlir::LogicalResult IDAInstance::addEquationsToIDA(
    mlir::OpBuilder& builder,
    mlir::Value idaInstance,
    const Model<ScheduledEquationsBlock>& model)
{
  mlir::Location loc = model.getOperation().getLoc();
  auto module = model.getOperation()->getParentOfType<mlir::ModuleOp>();

  // Substitute the accesses to non-IDA variables with the equations writing
  // in such variables.
  std::vector<std::unique_ptr<ScheduledEquation>> independentEquations;

  // First create the writes map, that is the knowledge of which equation
  // writes into a variable and in which indices.
  // The variables are mapped by their argument number.
  auto writesMap = getWritesMap(model);

  // The equations we are operating on
  std::queue<std::unique_ptr<ScheduledEquation>> processedEquations;

  for (const auto& equation : equations) {
    auto clone = Equation::build(
        equation->cloneIR(), equation->getVariables());

    auto matchedClone = std::make_unique<MatchedEquation>(
        std::move(clone),
        equation->getIterationRanges(),
        equation->getWrite().getPath());

    auto scheduledClone = std::make_unique<ScheduledEquation>(
        std::move(matchedClone),
        equation->getIterationRanges(),
        equation->getSchedulingDirection());

    processedEquations.push(std::move(scheduledClone));
  }

  while (!processedEquations.empty()) {
    auto& equation = processedEquations.front();
    IndexSet equationIndices = equation->getIterationRanges();

    bool atLeastOneAccessReplaced = false;

    for (const auto& access : equation->getReads()) {
      if (atLeastOneAccessReplaced) {
        // Avoid the duplicates.
        // For example, if we have the following equation
        //   eq1: z = x + y ...
        // and both x and y have to be replaced, then the replacement of 'x'
        // would create the equation
        //   eq2: z = f(...) + y ...
        // while the replacement of 'y' would create the equation
        //   eq3: z = x + g(...) ...
        // This implies that at the next round both would have respectively 'y'
        // and 'x' replaced for a second time, thus leading to two identical
        // equations:
        //   eq4: z = f(...) + g(...) ...
        //   eq5: z = f(...) + g(...) ...

        break;
      }

      auto readIndices = access.getAccessFunction().map(equationIndices);
      auto varPosition = access.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
      auto writingEquations = llvm::make_range(writesMap.equal_range(varPosition));

      for (const auto& entry : writingEquations) {
        ScheduledEquation* writingEquation = entry.second.second;

        if (equations.contains(writingEquation)) {
          // Ignore the equation if it is already managed by IDA.
          continue;
        }

        const IndexSet& writtenVariableIndices = entry.second.first;

        if (!writtenVariableIndices.overlaps(readIndices)) {
          continue;
        }

        atLeastOneAccessReplaced = true;

        auto clone = Equation::build(equation->cloneIR(), equation->getVariables());

        auto explicitWritingEquation = writingEquation->cloneIRAndExplicitate(builder);
        TemporaryEquationGuard guard(*explicitWritingEquation);

        IndexSet iterationRanges = explicitWritingEquation->getIterationRanges().getCanonicalRepresentation();

        for (const MultidimensionalRange& range : llvm::make_range(
                 iterationRanges.rangesBegin(),
                 iterationRanges.rangesEnd())) {
          if (mlir::failed(explicitWritingEquation->replaceInto(
                  builder, IndexSet(range), *clone,
                  access.getAccessFunction(), access.getPath()))) {
            return mlir::failure();
          }
        }

        // Add the equation with the replaced access
        IndexSet readAccessIndices = access.getAccessFunction().inverseMap(
            writtenVariableIndices,
            IndexSet(equationIndices));

        IndexSet newEquationIndices = readAccessIndices.intersect(equationIndices).getCanonicalRepresentation();

        for (const MultidimensionalRange& range : llvm::make_range(
                 newEquationIndices.rangesBegin(),
                 newEquationIndices.rangesEnd())) {
          auto matchedEquation = std::make_unique<MatchedEquation>(
              clone->clone(), IndexSet(range), equation->getWrite().getPath());

          auto scheduledEquation = std::make_unique<ScheduledEquation>(
              std::move(matchedEquation), IndexSet(range), equation->getSchedulingDirection());

          processedEquations.push(std::move(scheduledEquation));
        }
      }
    }

    if (atLeastOneAccessReplaced) {
      equation->eraseIR();
    } else {
      independentEquations.push_back(std::move(equation));
    }

    processedEquations.pop();
  }

  // Check that all the non-IDA variables have been replaced
  assert(llvm::all_of(independentEquations, [&](const auto& equation) {
           return llvm::all_of(equation->getAccesses(), [&](const Access& access) {
             mlir::Value variable = access.getVariable()->getValue();
             return hasVariable(variable);
           });
         }) && "Some non-IDA variables have not been replaced");

  // The accesses to non-IDA variables have been replaced. Now we can proceed
  // to create the residual and jacobian functions.

  // Counters used to obtain unique names for the functions.
  size_t residualFunctionsCounter = 0;
  size_t jacobianFunctionsCounter = 0;
  size_t partialDerTemplatesCounter = 0;

  llvm::DenseMap<unsigned int, mlir::Value> variablesMapping;

  for (const auto& [variable, idaVariable] : llvm::zip(algebraicVariables, idaAlgebraicVariables)) {
    unsigned int argNumber = variable.cast<mlir::BlockArgument>().getArgNumber();
    variablesMapping[argNumber] = idaVariable;
  }

  for (const auto& [variable, idaVariable] : llvm::zip(stateVariables, idaStateVariables)) {
    unsigned int argNumber = variable.cast<mlir::BlockArgument>().getArgNumber();
    variablesMapping[argNumber] = idaVariable;
  }

  for (const auto& [variable, idaVariable] : llvm::zip(derivativeVariables, idaStateVariables)) {
    unsigned int argNumber = variable.cast<mlir::BlockArgument>().getArgNumber();
    variablesMapping[argNumber] = idaVariable;
  }

  for (const auto& equation : independentEquations) {
    auto iterationRanges = equation->getIterationRanges(); //todo: check ragged case

    for(auto ranges : llvm::make_range(iterationRanges.rangesBegin(), iterationRanges.rangesEnd())) {
      std::vector<mlir::Attribute> rangesAttr;

      for (size_t i = 0; i < ranges.rank(); ++i) {
        rangesAttr.push_back(builder.getI64ArrayAttr({ ranges[i].getBegin(), ranges[i].getEnd() }));
      }

      auto writtenVarNumber = equation->getWrite().getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();

      auto idaEquation = builder.create<mlir::ida::AddEquationOp>(
          equation->getOperation().getLoc(),
          idaInstance,
          builder.getArrayAttr(rangesAttr),
          variablesMapping[writtenVarNumber],
          getAccessMap(builder, equation->getWrite().getAccessFunction()));

      if (mlir::failed(addVariableAccessesInfoToIDA(
              builder, idaInstance, *equation, idaEquation))) {
        return mlir::failure();
      }

      // Create the residual function
      std::string residualFunctionName = getUniqueSymbolName(module, [&]() {
        return "ida_residualFunction_" +
            std::to_string(residualFunctionsCounter++);
      });

      if (mlir::failed(createResidualFunction(
              builder, *equation, idaEquation, residualFunctionName))) {
        return mlir::failure();
      }

      builder.create<mlir::ida::SetResidualOp>(
          loc, idaInstance, idaEquation, residualFunctionName);

      // Create the partial derivative template.
      std::string partialDerTemplateName =
          getUniqueSymbolName(module, [&]() {
            return "ida_pder_" +
                std::to_string(partialDerTemplatesCounter++);
          });

      if (mlir::failed(createPartialDerTemplateFunction(
              builder, *equation, partialDerTemplateName))) {
        return mlir::failure();
      }

      // Create the Jacobian functions.
      // Notice that Jacobian functions are not created for parametric and
      // derivative variables. The former are in fact known by IDA just to
      // forward them for computations. The latter are already handled when
      // encountering the state variable through the 'alpha' parameter set
      // into the derivative seed.

      assert(algebraicVariables.size() == idaAlgebraicVariables.size());

      for (auto [variable, idaVariable] : llvm::zip(algebraicVariables, idaAlgebraicVariables)) {
        std::string jacobianFunctionName = getUniqueSymbolName(module, [&]() {
          return "ida_jacobianFunction_" +
              std::to_string(jacobianFunctionsCounter++);
        });

        if (mlir::failed(createJacobianFunction(
                builder, *equation, jacobianFunctionName, variable,
                partialDerTemplateName))) {
          return mlir::failure();
        }

        builder.create<mlir::ida::AddJacobianOp>(
            loc,
            idaInstance,
            idaEquation,
            idaVariable,
            jacobianFunctionName);
      }

      assert(stateVariables.size() == idaStateVariables.size());

      for (auto [variable, idaVariable] : llvm::zip(stateVariables, idaStateVariables)) {
        std::string jacobianFunctionName = getUniqueSymbolName(module, [&]() {
          return "ida_jacobianFunction_" +
              std::to_string(jacobianFunctionsCounter++);
        });

        if (mlir::failed(createJacobianFunction(
                builder, *equation, jacobianFunctionName, variable,
                partialDerTemplateName))) {
          return mlir::failure();
        }

        builder.create<mlir::ida::AddJacobianOp>(
            loc,
            idaInstance,
            idaEquation,
            idaVariable,
            jacobianFunctionName);
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult IDAInstance::addVariableAccessesInfoToIDA(
    mlir::OpBuilder& builder,
    mlir::Value idaInstance,
    const Equation& equation,
    mlir::Value idaEquation)
{
  assert(idaEquation.getType().isa<mlir::ida::EquationType>());

  mlir::Location loc = equation.getOperation().getLoc();

  auto getIDAVariable = [&](mlir::Value variable) -> mlir::Value {
    unsigned int argNumber =
        variable.cast<mlir::BlockArgument>().getArgNumber();

    if (derivativesMap->isDerivative(argNumber)) {
      unsigned int stateArgNumber =
          derivativesMap->getDerivedVariable(argNumber);

      assert(stateVariablesLookup.find(stateArgNumber) !=
             stateVariablesLookup.end());

      return idaStateVariables[stateVariablesLookup[stateArgNumber]];
    }

    if (derivativesMap->hasDerivative(argNumber)) {
      assert(stateVariablesLookup.find(argNumber) !=
             stateVariablesLookup.end());

      return idaStateVariables[stateVariablesLookup[argNumber]];
    }

    assert(algebraicVariablesLookup.find(argNumber) !=
           algebraicVariablesLookup.end());

    return idaAlgebraicVariables[algebraicVariablesLookup[argNumber]];
  };

  // Keep track of the discovered accesses in order to avoid adding the same
  // access map multiple times for the same variable.
  llvm::DenseMap<mlir::Value, llvm::DenseSet<mlir::AffineMap>> maps;

  for (const Access& access : equation.getAccesses()) {
    mlir::Value variable = access.getVariable()->getValue();

    // Skip parametric variables. They are used only as read-only values.
    if (hasParametricVariable(variable)) {
      continue;
    }

    mlir::Value idaVariable = getIDAVariable(variable);

    const auto& accessFunction = access.getAccessFunction();
    std::vector<mlir::AffineExpr> expressions;

    for (const auto& dimensionAccess : accessFunction) {
      if (dimensionAccess.isConstantAccess()) {
        expressions.push_back(mlir::getAffineConstantExpr(
            dimensionAccess.getPosition(), builder.getContext()));
      } else {
        auto baseAccess = mlir::getAffineDimExpr(
            dimensionAccess.getInductionVariableIndex(),
            builder.getContext());

        auto withOffset = baseAccess + dimensionAccess.getOffset();
        expressions.push_back(withOffset);
      }
    }

    auto affineMap = mlir::AffineMap::get(
        accessFunction.size(), 0, expressions, builder.getContext());

    assert(idaVariable != nullptr);
    maps[idaVariable].insert(affineMap);
  }

  // Inform IDA about the discovered accesses.
  for (const auto& entry : maps) {
    mlir::Value idaVariable = entry.getFirst();

    for (const auto& map : entry.getSecond()) {
      builder.create<mlir::ida::AddVariableAccessOp>(
          loc, idaInstance, idaEquation, idaVariable, map);
    }
  }

  return mlir::success();
}

mlir::LogicalResult IDAInstance::createResidualFunction(
    mlir::OpBuilder& builder,
    const Equation& equation,
    mlir::Value idaEquation,
    llvm::StringRef residualFunctionName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = equation.getOperation().getLoc();

  // Add the function to the end of the module.
  auto module = equation.getOperation()->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(module.getBody());

  // Create the function.
  std::vector<mlir::Value> managedVariables = getIDAFunctionArgs();

  auto residualFunction = builder.create<mlir::ida::ResidualFunctionOp>(
      loc,
      residualFunctionName,
      RealType::get(builder.getContext()),
      mlir::ValueRange(managedVariables).getTypes(),
      equation.getNumOfIterationVars(),
      RealType::get(builder.getContext()));

  mlir::Block* bodyBlock = residualFunction.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);

  // Map the original variables to the ones received by the function, which
  // are in a possibly different order.
  mlir::BlockAndValueMapping mapping;

  for (auto original : llvm::enumerate(managedVariables)) {
    mlir::Value mapped = residualFunction.getVariables()[original.index()];
    mapping.map(original.value(), mapped);
  }

  // Map the iteration variables.
  auto originalInductions = equation.getInductionVariables();
  auto mappedInductions = residualFunction.getEquationIndices();

  // Scalar equations have zero concrete values, but yet they show a fake
  // induction variable. The same happens with equations having implicit
  // iteration variables (which originate from array assignments).
  assert(originalInductions.size() <= mappedInductions.size());

  for (size_t i = 0; i < originalInductions.size(); ++i) {
    mapping.map(originalInductions[i], mappedInductions[i]);
  }

  for (auto& op : equation.getOperation().bodyBlock()->getOperations()) {
    if (auto timeOp = mlir::dyn_cast<TimeOp>(op)) {
      mapping.map(timeOp.getResult(), residualFunction.getArguments()[0]);
    } else {
      builder.clone(op, mapping);
    }
  }

  // Create the instructions to compute the difference between the right-hand
  // side and the left-hand side of the equation.
  auto clonedTerminator = mlir::cast<EquationSidesOp>(
      residualFunction.getBodyRegion().back().getTerminator());

  assert(clonedTerminator.getLhsValues().size() == 1);
  assert(clonedTerminator.getRhsValues().size() == 1);

  mlir::Value lhs = clonedTerminator.getLhsValues()[0];
  mlir::Value rhs = clonedTerminator.getRhsValues()[0];

  if (lhs.getType().isa<ArrayType>()) {
    std::vector<mlir::Value> indices(
        std::next(mappedInductions.begin(), originalInductions.size()),
        mappedInductions.end());

    lhs = builder.create<LoadOp>(lhs.getLoc(), lhs, indices);

    assert((lhs.getType().isa<
            mlir::IndexType, BooleanType, IntegerType, RealType>()));
  }

  if (rhs.getType().isa<ArrayType>()) {
    std::vector<mlir::Value> indices(
        std::next(mappedInductions.begin(), originalInductions.size()),
        mappedInductions.end());

    rhs = builder.create<LoadOp>(rhs.getLoc(), rhs, indices);

    assert((rhs.getType().isa<
            mlir::IndexType, BooleanType, IntegerType, RealType>()));
  }

  mlir::Value difference = builder.create<SubOp>(
      loc, RealType::get(builder.getContext()), rhs, lhs);

  builder.create<mlir::ida::ReturnOp>(difference.getLoc(), difference);

  // Erase the old terminator.
  auto lhsOp = clonedTerminator.getLhs().getDefiningOp<EquationSideOp>();
  auto rhsOp = clonedTerminator.getRhs().getDefiningOp<EquationSideOp>();

  clonedTerminator.erase();

  lhsOp.erase();
  rhsOp.erase();

  return mlir::success();
}

mlir::LogicalResult IDAInstance::createPartialDerTemplateFunction(
    mlir::OpBuilder& builder,
    const Equation& equation,
    llvm::StringRef templateName)
{
  mlir::Location loc = equation.getOperation().getLoc();

  auto partialDerTemplate = createPartialDerTemplateFromEquation(
      builder, equation, templateName);

  // Add the time to the input members (and signature).
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(partialDerTemplate.bodyBlock());

  auto timeMember = builder.create<MemberCreateOp>(
      loc,
      "time",
      MemberType::get(
          llvm::None,
          RealType::get(builder.getContext()), false, IOProperty::input),
      llvm::None);

  mlir::Value time = builder.create<MemberLoadOp>(loc, timeMember);

  std::vector<mlir::Type> args;
  args.push_back(timeMember.getMemberType().unwrap());

  for (auto type : partialDerTemplate.getFunctionType().getInputs()) {
    args.push_back(type);
  }

  partialDerTemplate->setAttr(
      partialDerTemplate.getFunctionTypeAttrName(),
      mlir::TypeAttr::get(builder.getFunctionType(
          args, partialDerTemplate.getFunctionType().getResults())));

  // Replace the TimeOp with the newly created member.
  partialDerTemplate.walk([&](TimeOp timeOp) {
    timeOp.replaceAllUsesWith(time);
    timeOp.erase();
  });

  return mlir::success();
}

FunctionOp IDAInstance::createPartialDerTemplateFromEquation(
    mlir::OpBuilder& builder,
    const Equation& equation,
    llvm::StringRef templateName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = equation.getOperation().getLoc();

  // Add the function to the end of the module.
  auto module = equation.getOperation()->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(module.getBody());

  // Create the function.
  std::string functionOpName = templateName.str() + "_base";
  std::vector<mlir::Value> managedVariables = getIDAFunctionArgs();

  // The arguments of the base function contain both the variables and the
  // inductions.
  llvm::SmallVector<mlir::Type> argsTypes;

  for (auto type : mlir::ValueRange(managedVariables).getTypes()) {
    if (auto arrayType = type.dyn_cast<ArrayType>();
        arrayType && arrayType.isScalar()) {
      argsTypes.push_back(arrayType.getElementType());
    } else {
      argsTypes.push_back(type);
    }
  }

  for (size_t i = 0; i < equation.getNumOfIterationVars(); ++i) {
    argsTypes.push_back(builder.getIndexType());
  }

  // Create the function to be derived.
  auto functionType = builder.getFunctionType(
      argsTypes, RealType::get(builder.getContext()));

  auto functionOp = builder.create<FunctionOp>(
      loc, functionOpName, functionType);

  // Start the body of the function.
  mlir::Block* entryBlock = builder.createBlock(&functionOp.getBody());
  builder.setInsertionPointToStart(entryBlock);

  // Create the input members and map them to the original variables (and
  // inductions).
  mlir::BlockAndValueMapping mapping;

  for (auto originalVar : llvm::enumerate(managedVariables)) {
    std::string memberName = "var" + std::to_string(originalVar.index());

    auto memberType = MemberType::wrap(
        originalVar.value().getType(), false, IOProperty::input);

    auto memberOp = builder.create<MemberCreateOp>(
        loc, memberName, memberType, llvm::None);

    mapping.map(originalVar.value(), memberOp);
  }

  llvm::SmallVector<mlir::Value, 3> inductions;

  for (size_t i = 0; i < equation.getNumOfIterationVars(); ++i) {
    std::string memberName = "ind" + std::to_string(i);

    auto memberType = MemberType::wrap(
        builder.getIndexType(), false, IOProperty::input);

    auto memberOp = builder.create<MemberCreateOp>(
        loc, memberName, memberType, llvm::None);

    inductions.push_back(memberOp);
  }

  // Create the output member, that is the difference between its equation
  // right-hand side value and its left-hand side value.
  auto originalTerminator = mlir::cast<EquationSidesOp>(
      equation.getOperation().bodyBlock()->getTerminator());

  assert(originalTerminator.getLhsValues().size() == 1);
  assert(originalTerminator.getRhsValues().size() == 1);

  auto outputMember = builder.create<MemberCreateOp>(
      loc, "out",
      MemberType::wrap(
          RealType::get(builder.getContext()),
          false,
          IOProperty::output),
      llvm::None);

  // Now that all the members have been created, we can load the input
  // members and the inductions.
  for (mlir::Value originalVar : managedVariables) {
    auto mappedVar = builder.create<MemberLoadOp>(
        loc, mapping.lookup(originalVar));

    mapping.map(originalVar, mappedVar);
  }

  for (mlir::Value& induction : inductions) {
    induction = builder.create<MemberLoadOp>(loc, induction);
  }

  auto explicitEquationInductions = equation.getInductionVariables();

  for (const auto& originalInduction : llvm::enumerate(explicitEquationInductions)) {
    assert(originalInduction.index() < inductions.size());
    mapping.map(originalInduction.value(), inductions[originalInduction.index()]);
  }

  // Clone the original operations.

  for (auto& op : equation.getOperation().bodyBlock()->getOperations()) {
    if (auto loadOp = mlir::dyn_cast<LoadOp>(op)) {
      // We need to check if the load operation is performed on a model
      // variable. Those variables are in fact created inside the function by
      // means of MemberCreateOps, and if such variable is a scalar one there
      // would be a load operation wrongly operating on a scalar value.

      if (loadOp.getArray().getType().cast<ArrayType>().isScalar()) {
        mlir::Value mapped = mapping.lookup(mlir::Value(loadOp.getArray()));

        if (auto memberLoadOp = mapped.getDefiningOp<MemberLoadOp>()) {
          mapping.map(loadOp.getResult(), memberLoadOp.getResult());
          continue;
        }
      }
    }

    builder.clone(op, mapping);
  }

  auto terminator = mlir::cast<EquationSidesOp>(
      functionOp.bodyBlock()->getTerminator());

  assert(terminator.getLhsValues().size() == 1);
  assert(terminator.getRhsValues().size() == 1);

  mlir::Value lhs = terminator.getLhsValues()[0];
  mlir::Value rhs = terminator.getRhsValues()[0];

  if (auto arrayType = lhs.getType().dyn_cast<ArrayType>()) {
    assert(rhs.getType().isa<ArrayType>());
    assert(arrayType.getRank() + explicitEquationInductions.size() == inductions.size());
    auto implicitInductions = llvm::makeArrayRef(inductions).take_back(arrayType.getRank());

    lhs = builder.create<LoadOp>(loc, lhs, implicitInductions);
    rhs = builder.create<LoadOp>(loc, rhs, implicitInductions);
  }

  auto result = builder.create<SubOp>(
      loc, RealType::get(builder.getContext()), rhs, lhs);

  builder.create<MemberStoreOp>(loc, outputMember, result);

  auto lhsOp = terminator.getLhs().getDefiningOp<EquationSideOp>();
  auto rhsOp = terminator.getRhs().getDefiningOp<EquationSideOp>();

  terminator.erase();

  lhsOp.erase();
  rhsOp.erase();

  // Create the derivative template function.
  ForwardAD forwardAD;

  auto derTemplate = forwardAD.createPartialDerTemplateFunction(
      builder, loc, functionOp, templateName);

  functionOp.erase();

  return derTemplate;
}

mlir::LogicalResult IDAInstance::createJacobianFunction(
    mlir::OpBuilder& builder,
    const Equation& equation,
    llvm::StringRef jacobianFunctionName,
    mlir::Value independentVariable,
    llvm::StringRef partialDerTemplateName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = equation.getOperation().getLoc();

  // Add the function to the end of the module.
  auto module = equation.getOperation()->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(module.getBody());

  // Create the function.
  std::vector<mlir::Value> managedVariables = getIDAFunctionArgs();

  auto jacobianFunction = builder.create<mlir::ida::JacobianFunctionOp>(
      loc,
      jacobianFunctionName,
      RealType::get(builder.getContext()),
      mlir::ValueRange(managedVariables).getTypes(),
      equation.getNumOfIterationVars(),
      independentVariable.getType().cast<ArrayType>().getRank(),
      RealType::get(builder.getContext()),
      RealType::get(builder.getContext()));

  mlir::Block* bodyBlock = jacobianFunction.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);

  // List of the arguments to be passed to the derivative template function.
  llvm::SmallVector<mlir::Value> args;

  // Keep track of the seeds consisting in arrays, so that we can deallocate
  // when not being used anymore.
  llvm::SmallVector<mlir::Value> arraySeeds;

  mlir::Value zero = builder.create<ConstantOp>(
      loc, RealAttr::get(builder.getContext(), 0));

  mlir::Value one = builder.create<ConstantOp>(
      loc, RealAttr::get(builder.getContext(), 1));

  auto populateArgsFn =
      [&](unsigned int independentVarArgNumber,
          llvm::SmallVectorImpl<mlir::Value>& args,
          llvm::SmallVectorImpl<mlir::Value>& seedArrays) {
        // 'Time' variable.
        args.push_back(jacobianFunction.getTime());

        // The variables.
        for (const auto& var : jacobianFunction.getVariables()) {
          if (auto arrayType = var.getType().dyn_cast<ArrayType>();
              arrayType && arrayType.isScalar()) {
            args.push_back(builder.create<LoadOp>(loc, var));
          } else {
            args.push_back(var);
          }
        }

        // Equation indices.
        for (auto equationIndex : jacobianFunction.getEquationIndices()) {
          args.push_back(equationIndex);
        }

        unsigned int oneSeedPosition = independentVarArgNumber;
        llvm::Optional<unsigned int> alphaSeedPosition = llvm::None;

        if (cleverDAE && derivativesMap->hasDerivative(independentVarArgNumber)) {
          alphaSeedPosition =
              derivativesMap->getDerivative(independentVarArgNumber);
        }

        // Create the seed values for the variables.
        for (mlir::Value var : managedVariables) {
          auto varArgNumber = var.cast<mlir::BlockArgument>().getArgNumber();

          if (auto arrayType = var.getType().dyn_cast<ArrayType>();
              arrayType && !arrayType.isScalar()) {
            assert(arrayType.hasStaticShape());

            auto array = builder.create<ArrayBroadcastOp>(
                loc,
                arrayType.toElementType(RealType::get(builder.getContext())),
                zero);

            seedArrays.push_back(array);
            args.push_back(array);

            if (varArgNumber == oneSeedPosition) {
              builder.create<StoreOp>(
                  loc, one, array,
                  jacobianFunction.getVariableIndices());

            } else if (alphaSeedPosition.has_value() && varArgNumber == *alphaSeedPosition) {
              builder.create<StoreOp>(
                  loc,
                  jacobianFunction.getAlpha(),
                  array,
                  jacobianFunction.getVariableIndices());
            }
          } else {
            assert(arrayType && arrayType.isScalar());

            if (varArgNumber == oneSeedPosition) {
              args.push_back(one);
            } else if (alphaSeedPosition.has_value() && varArgNumber == *alphaSeedPosition) {
              args.push_back(jacobianFunction.getAlpha());
            } else {
              args.push_back(zero);
            }
          }
        }

        // Seeds of the equation indices. They are all equal to zero.
        for (size_t i = 0; i < jacobianFunction.getEquationIndices().size(); ++i) {
          args.push_back(zero);
        }
      };

  // Derivative with respect to the algebraic / state variable.
  unsigned int independentVarArgNumber =
      independentVariable.cast<mlir::BlockArgument>().getArgNumber();

  populateArgsFn(independentVarArgNumber, args, arraySeeds);

  // Call the derivative template.
  auto templateCall = builder.create<CallOp>(
      loc,
      partialDerTemplateName,
      RealType::get(builder.getContext()),
      args);

  mlir::Value result = templateCall.getResult(0);

  // Deallocate the seeds consisting in arrays.
  for (mlir::Value seed : arraySeeds) {
    builder.create<FreeOp>(loc, seed);
  }

  if (!cleverDAE && derivativesMap->hasDerivative(independentVarArgNumber)) {
    args.clear();
    arraySeeds.clear();

    populateArgsFn(
        derivativesMap->getDerivative(independentVarArgNumber),
        args, arraySeeds);

    auto secondTemplateCall = builder.create<CallOp>(
        loc,
        partialDerTemplateName,
        RealType::get(builder.getContext()),
        args);

    for (mlir::Value seed : arraySeeds) {
      builder.create<FreeOp>(loc, seed);
    }

    mlir::Value secondDerivativeTimesAlpha = builder.create<MulOp>(
        loc, RealType::get(builder.getContext()),
        jacobianFunction.getAlpha(), secondTemplateCall.getResult(0));

    result = builder.create<AddOp>(
        loc, RealType::get(builder.getContext()),
        result, secondDerivativeTimesAlpha);
  }

  builder.create<mlir::ida::ReturnOp>(loc, result);

  return mlir::success();
}

mlir::LogicalResult IDAInstance::performCalcIC(
    mlir::OpBuilder& builder,
    mlir::Value idaInstance)
{
  builder.create<mlir::ida::CalcICOp>(idaInstance.getLoc(), idaInstance);
  return mlir::success();
}

mlir::LogicalResult IDAInstance::performStep(
    mlir::OpBuilder& builder,
    mlir::Value idaInstance)
{
  builder.create<mlir::ida::StepOp>(idaInstance.getLoc(), idaInstance);
  return mlir::success();
}

mlir::Value IDAInstance::getCurrentTime(
    mlir::OpBuilder& builder,
    mlir::Value idaInstance,
    mlir::Type timeType)
{
  return builder.create<mlir::ida::GetCurrentTimeOp>(
      idaInstance.getLoc(), timeType, idaInstance);
}

std::vector<mlir::Value> IDAInstance::getIDAFunctionArgs() const
{
  std::vector<mlir::Value> result;

  // Add the parametric variables.
  for (mlir::Value variable : parametricVariables) {
    result.push_back(variable);
  }

  // Add the algebraic variables.
  for (mlir::Value variable : algebraicVariables) {
    result.push_back(variable);
  }

  // Add the state variables.
  for (mlir::Value variable : stateVariables) {
    result.push_back(variable);
  }

  // Add the derivative variables.
  // The derivatives must be in the same order of the respective state
  // variables.
  llvm::DenseMap<unsigned int, size_t> derPosMap;

  for (auto variable : llvm::enumerate(derivativeVariables)) {
    auto argNumber =
        variable.value().cast<mlir::BlockArgument>().getArgNumber();

    derPosMap[argNumber] = variable.index();
  }

  for (mlir::Value stateVariable : stateVariables) {
    auto stateArgNumber =
        stateVariable.cast<mlir::BlockArgument>().getArgNumber();

    auto derivativeArgNumber = derivativesMap->getDerivative(stateArgNumber);
    assert(derPosMap.find(derivativeArgNumber) != derPosMap.end());
    size_t derivativePosition = derPosMap[derivativeArgNumber];
    result.push_back(derivativeVariables[derivativePosition]);
  }

  return result;
}

std::multimap<unsigned int, std::pair<IndexSet, ScheduledEquation*>>
IDAInstance::getWritesMap(const Model<ScheduledEquationsBlock>& model) const
{
  std::multimap<unsigned int, std::pair<IndexSet, ScheduledEquation*>> writesMap;

  for (const auto& equationsBlock : model.getScheduledBlocks()) {
    for (const auto& equation : *equationsBlock) {
      const Access& write = equation->getWrite();

      mlir::Value writtenVariable = write.getVariable()->getValue();

      auto varPosition =
          writtenVariable.cast<mlir::BlockArgument>().getArgNumber();

      IndexSet writtenIndices =
          write.getAccessFunction().map(equation->getIterationRanges());

      writesMap.emplace(
          varPosition,
          std::make_pair(writtenIndices, equation.get()));
    }
  }

  return writesMap;
}

mlir::AffineMap IDAInstance::getAccessMap(
    mlir::OpBuilder& builder,
    const AccessFunction& accessFunction) const
{
  std::vector<mlir::AffineExpr> expressions;

  for (const auto& dimensionAccess : accessFunction) {
    if (dimensionAccess.isConstantAccess()) {
      expressions.push_back(mlir::getAffineConstantExpr(
          dimensionAccess.getPosition(), builder.getContext()));
    } else {
      auto baseAccess = mlir::getAffineDimExpr(
          dimensionAccess.getInductionVariableIndex(),
          builder.getContext());

      auto withOffset = baseAccess + dimensionAccess.getOffset();
      expressions.push_back(withOffset);
    }
  }

  return mlir::AffineMap::get(
      accessFunction.size(), 0, expressions, builder.getContext());
}

namespace
{
  class IDASolver : public mlir::modelica::impl::ModelSolver
  {
    public:
      using ExplicitEquationsMap =
        llvm::DenseMap<ScheduledEquation*, Equation*>;

      IDASolver(bool cleverDAE);

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
      /// Create the function that instantiates the external solvers to be used
      /// during the IC computation.
      mlir::LogicalResult createInitICSolversFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          const Model<ScheduledEquationsBlock>& model,
          IDAInstance* idaInstance) const;

      /// Create the function that deallocates the external solvers used during
      /// the IC computation.
      mlir::LogicalResult createDeinitICSolversFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          IDAInstance* idaInstance) const;

      /// Create the function that instantiates the external solvers to be used
      /// during the simulation.
      mlir::LogicalResult createInitMainSolversFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          const Model<ScheduledEquationsBlock>& model,
          IDAInstance* idaInstance) const;

      /// Create the function that deallocates the external solvers used during
      /// the simulation.
      mlir::LogicalResult createDeinitMainSolversFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          IDAInstance* idaInstance) const;

      /// Create the function that instantiates the external solvers.
      mlir::LogicalResult createInitSolversFunction(
          mlir::OpBuilder& builder,
          mlir::ValueRange variables,
          const Model<ScheduledEquationsBlock>& model,
          IDAInstance* idaInstance) const;

      /// Create the function that deallocates the external solvers.
      mlir::LogicalResult createDeinitSolversFunction(
          mlir::OpBuilder& builder,
          mlir::Value instance,
          IDAInstance* idaInstance) const;

      /// Create the function that computes the initial conditions of the
      /// "initial conditions model".
      mlir::LogicalResult createSolveICModelFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          const Model<ScheduledEquationsBlock>& model,
          const ExplicitEquationsMap& explicitEquationsMap,
          IDAInstance* idaInstance) const;

      /// Create the function that computes the initial conditions of the "main
      /// model".
      mlir::LogicalResult createCalcICFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          const Model<ScheduledEquationsBlock>& model,
          IDAInstance* idaInstance) const;

      /// Create the functions that calculates the values that the variables
      /// belonging to IDA will have in the next iteration.
      mlir::LogicalResult createUpdateIDAVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          const Model<ScheduledEquationsBlock>& model,
          IDAInstance* idaInstance) const;

      /// Create the functions that calculates the values that the variables
      /// not belonging to IDA will have in the next iteration.
      mlir::LogicalResult createUpdateNonIDAVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          const Model<ScheduledEquationsBlock>& model,
          const ExplicitEquationsMap& explicitEquationsMap,
          IDAInstance* idaInstance) const;

      /// Create the function to be used to get the time reached by IDA.
      mlir::LogicalResult createGetIDATimeFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          const Model<ScheduledEquationsBlock>& model,
          IDAInstance* idaInstance) const;

      mlir::func::FuncOp createEquationFunction(
          mlir::OpBuilder& builder,
          const ScheduledEquation& equation,
          llvm::StringRef equationFunctionName,
          mlir::func::FuncOp templateFunction,
          mlir::TypeRange varsTypes) const;

    private:
      bool cleverDAE;
  };
}

/// Get or create the template equation function for a scheduled equation.
static llvm::Optional<EquationTemplate> getOrCreateEquationTemplateFunction(
    llvm::ThreadPool& threadPool,
    mlir::OpBuilder& builder,
    ModelOp modelOp,
    const ScheduledEquation* equation,
    const IDASolver::ExplicitEquationsMap& explicitEquationsMap,
    std::map<EquationTemplateInfo, EquationTemplate>& equationTemplatesMap,
    llvm::StringRef functionNamePrefix,
    size_t& equationTemplateCounter)
{
  EquationInterface equationInt = equation->getOperation();

  EquationTemplateInfo requestedTemplate(
      equationInt, equation->getSchedulingDirection());

  // Check if the template equation already exists.
  if (auto it = equationTemplatesMap.find(requestedTemplate);
      it != equationTemplatesMap.end()) {
    return it->second;
  }

  // If not, create it.
  mlir::Location loc = equationInt.getLoc();

  // Name of the function.
  std::string functionName = functionNamePrefix.str() +
      std::to_string(equationTemplateCounter++);

  auto explicitEquation = llvm::find_if(
      explicitEquationsMap,
      [&](const auto& equationPtr) {
        return equationPtr.first == equation;
      });

  if (explicitEquation == explicitEquationsMap.end()) {
    // The equation can't be made explicit and is not passed to any external solver
    return llvm::None;
  }

  // Create the equation template function
  std::vector<unsigned int> usedVariables;

  mlir::func::FuncOp function =
      explicitEquation->second->createTemplateFunction(
          threadPool, builder, functionName,
          equation->getSchedulingDirection(),
          usedVariables);

  size_t timeArgumentIndex = equation->getNumOfIterationVars() * 3;

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

IDASolver::IDASolver(bool cleverDAE)
    : cleverDAE(cleverDAE)
{
}

mlir::LogicalResult IDASolver::solveICModel(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    const Model<ScheduledEquationsBlock>& model)
{
  DerivativesMap emptyDerivativesMap;

  auto idaInstance = std::make_unique<IDAInstance>(
      emptyDerivativesMap, cleverDAE);

  idaInstance->setStartTime(0);
  idaInstance->setEndTime(0);

  std::set<std::unique_ptr<Equation>> explicitEquationsStorage;
  ExplicitEquationsMap explicitEquationsMap;

  if (cleverDAE) {
    llvm::DenseSet<ScheduledEquation*> implicitEquations;
    llvm::DenseSet<ScheduledEquation*> cyclicEquations;

    // Determine which equations can be potentially processed by just making
    // them explicit with respect to the variable they match.

    for (auto& scheduledBlock : model.getScheduledBlocks()) {
      if (!scheduledBlock->hasCycle()) {
        for (auto& scheduledEquation : *scheduledBlock) {
          auto explicitClone =
              scheduledEquation->cloneIRAndExplicitate(builder);

          if (explicitClone == nullptr) {
            implicitEquations.insert(scheduledEquation.get());
          } else {
            auto movedClone =
                explicitEquationsStorage.emplace(std::move(explicitClone));

            assert(movedClone.second);
            explicitEquationsMap[scheduledEquation.get()] =
                movedClone.first->get();
          }
        }
      } else {
        for (const auto& equation : *scheduledBlock) {
          cyclicEquations.insert(equation.get());
        }
      }
    }

    // Add the implicit equations to the set of equations managed by IDA,
    // together with their written variables.

    for (ScheduledEquation* implicitEquation : implicitEquations) {
      idaInstance->addEquation(implicitEquation);

      mlir::Value var = implicitEquation->getWrite().getVariable()->getValue();
      idaInstance->addAlgebraicVariable(var);
    }

    // Add the cyclic equations to the set of equations managed by IDA,
    // together with their written variables.

    for (ScheduledEquation* cyclicEquation : cyclicEquations) {
      idaInstance->addEquation(cyclicEquation);

      mlir::Value var = cyclicEquation->getWrite().getVariable()->getValue();
      idaInstance->addAlgebraicVariable(var);
    }

    // If any of the remaining equations manageable by MARCO does write on a
    // variable managed by IDA, then the equation must be passed to IDA even
    // if the scalar variables that are written do not belong to IDA.
    // Avoiding this would require either memory duplication or a more severe
    // restructuring of the solving infrastructure, which would have to be
    // able to split variables and equations according to which runtime
    // solver manages such variables.

    for (const auto& scheduledBlock : model.getScheduledBlocks()) {
      for (auto& scheduledEquation : *scheduledBlock) {
        mlir::Value var =
            scheduledEquation->getWrite().getVariable()->getValue();

        if (idaInstance->hasVariable(var)) {
          idaInstance->addEquation(scheduledEquation.get());
        }
      }
    }

    // Add the used parameters.

    for (const auto& scheduledBlock : model.getScheduledBlocks()) {
      for (const auto& equation : *scheduledBlock) {
        for (const Access& access : equation->getAccesses()) {
          if (auto var = access.getVariable(); var->isConstant()) {
            idaInstance->addParametricVariable(var->getValue());
          }
        }
      }
    }
  } else {
    for (const auto& scheduledBlock : model.getScheduledBlocks()) {
      // Add all the equations and all the written variables.
      for (const auto& equation : *scheduledBlock) {
        idaInstance->addEquation(equation.get());

        mlir::Value var = equation->getWrite().getVariable()->getValue();
        idaInstance->addAlgebraicVariable(var);

        // Also add the used parameters.
        for (const Access& access : equation->getAccesses()) {
          if (auto accessVar = access.getVariable();
              accessVar->isConstant()) {
            idaInstance->addParametricVariable(accessVar->getValue());
          }
        }
      }
    }
  }

  if (mlir::failed(createInitICSolversFunction(
          builder, simulationModuleOp, model, idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createDeinitICSolversFunction(
          builder, simulationModuleOp, idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createSolveICModelFunction(
          builder, simulationModuleOp, model,
          explicitEquationsMap, idaInstance.get()))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult IDASolver::solveMainModel(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    const Model<ScheduledEquationsBlock>& model)
{
  const DerivativesMap& derivativesMap = model.getDerivativesMap();
  auto idaInstance = std::make_unique<IDAInstance>(derivativesMap, cleverDAE);

  std::set<std::unique_ptr<Equation>> explicitEquationsStorage;
  ExplicitEquationsMap explicitEquationsMap;

  if (cleverDAE) {
    llvm::DenseSet<ScheduledEquation*> implicitEquations;
    llvm::DenseSet<ScheduledEquation*> cyclicEquations;

    // Determine which equations can be potentially processed by MARCO.
    // Those are the ones that can me bade explicit with respect to the
    // matched variable and the non-cyclic ones.

    for (auto& scheduledBlock : model.getScheduledBlocks()) {
      if (!scheduledBlock->hasCycle()) {
        for (auto& scheduledEquation : *scheduledBlock) {
          auto explicitClone =
              scheduledEquation->cloneIRAndExplicitate(builder);

          if (explicitClone == nullptr) {
            implicitEquations.insert(scheduledEquation.get());
          } else {
            auto movedClone =
                explicitEquationsStorage.emplace(std::move(explicitClone));

            assert(movedClone.second);
            explicitEquationsMap[scheduledEquation.get()] =
                movedClone.first->get();
          }
        }
      } else {
        for (const auto& equation : *scheduledBlock) {
          cyclicEquations.insert(equation.get());
        }
      }
    }

    // Add the implicit equations to the set of equations managed by IDA,
    // together with their written variables.

    for (ScheduledEquation* implicitEquation : implicitEquations) {
      idaInstance->addEquation(implicitEquation);

      mlir::Value var =
          implicitEquation->getWrite().getVariable()->getValue();

      unsigned int argNumber =
          var.cast<mlir::BlockArgument>().getArgNumber();

      if (derivativesMap.isDerivative(argNumber)) {
        idaInstance->addDerivativeVariable(var);
      } else if (derivativesMap.hasDerivative(argNumber)) {
        idaInstance->addStateVariable(var);
      } else {
        idaInstance->addAlgebraicVariable(var);
      }
    }

    // Add the cyclic equations to the set of equations managed by IDA,
    // together with their written variables.

    for (ScheduledEquation* cyclicEquation : cyclicEquations) {
      idaInstance->addEquation(cyclicEquation);

      mlir::Value var = cyclicEquation->getWrite().getVariable()->getValue();

      unsigned int argNumber =
          var.cast<mlir::BlockArgument>().getArgNumber();

      if (derivativesMap.isDerivative(argNumber)) {
        idaInstance->addDerivativeVariable(var);
      } else if (derivativesMap.hasDerivative(argNumber)) {
        idaInstance->addStateVariable(var);
      } else {
        idaInstance->addAlgebraicVariable(var);
      }
    }

    // Add the differential equations (i.e. the ones matched with a
    // derivative) to the set of equations managed by IDA, together with
    // their written variables.

    for (const auto& scheduledBlock : model.getScheduledBlocks()) {
      for (auto& scheduledEquation : *scheduledBlock) {
        mlir::Value var =
            scheduledEquation->getWrite().getVariable()->getValue();

        auto argNumber = var.cast<mlir::BlockArgument>().getArgNumber();

        if (derivativesMap.isDerivative(argNumber)) {
          idaInstance->addEquation(scheduledEquation.get());

          // State variable.
          mlir::Value stateVar =
              model.getOperation().getBodyRegion().getArgument(
                  derivativesMap.getDerivedVariable(argNumber));

          idaInstance->addStateVariable(stateVar);

          // Derivative variable.
          idaInstance->addDerivativeVariable(var);
        }
      }
    }

    // If any of the remaining equations manageable by MARCO does write on a
    // variable managed by IDA, then the equation must be passed to IDA even
    // if not strictly necessary. Avoiding this would require either memory
    // duplication or a more severe restructuring of the solving
    // infrastructure, which would have to be able to split variables and
    // equations according to which runtime solver manages such variables.

    for (const auto& scheduledBlock : model.getScheduledBlocks()) {
      for (auto& scheduledEquation : *scheduledBlock) {
        mlir::Value var =
            scheduledEquation->getWrite().getVariable()->getValue();

        if (idaInstance->hasVariable(var)) {
          idaInstance->addEquation(scheduledEquation.get());
        }
      }
    }

    // Add the used parameters.

    for (const auto& scheduledBlock : model.getScheduledBlocks()) {
      for (auto& equation : *scheduledBlock) {
        for (const Access& access : equation->getAccesses()) {
          const Variable* variable = access.getVariable();

          if (variable->isConstant()) {
            idaInstance->addParametricVariable(variable->getValue());
          }
        }
      }
    }
  } else {
    for (const auto& scheduledBlock : model.getScheduledBlocks()) {
      // Add all the equations and all the written variables.
      for (const auto& equation : *scheduledBlock) {
        idaInstance->addEquation(equation.get());

        mlir::Value var = equation->getWrite().getVariable()->getValue();

        unsigned int argNumber =
            var.cast<mlir::BlockArgument>().getArgNumber();

        if (derivativesMap.isDerivative(argNumber)) {
          // State variable.
          mlir::Value stateVar =
              model.getOperation().getBodyRegion().getArgument(
                  derivativesMap.getDerivedVariable(argNumber));

          idaInstance->addStateVariable(stateVar);

          // Derivative variable.
          idaInstance->addDerivativeVariable(var);
        } else if (derivativesMap.hasDerivative(argNumber)) {
          // State variable.
          idaInstance->addStateVariable(var);

          // Derivative variable.
          mlir::Value derVar =
              model.getOperation().getBodyRegion().getArgument(
                  derivativesMap.getDerivative(argNumber));

          idaInstance->addDerivativeVariable(derVar);
        } else {
          idaInstance->addAlgebraicVariable(var);
        }

        // Also add the used parameters.
        for (const Access& access : equation->getAccesses()) {
          if (auto accessVar = access.getVariable();
              accessVar->isConstant()) {
            idaInstance->addParametricVariable(accessVar->getValue());
          }
        }
      }
    }
  }

  if (mlir::failed(createInitMainSolversFunction(
          builder, simulationModuleOp, model, idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createDeinitMainSolversFunction(
          builder, simulationModuleOp, idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createCalcICFunction(
          builder, simulationModuleOp, model, idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateIDAVariablesFunction(
          builder, simulationModuleOp, model, idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateNonIDAVariablesFunction(
          builder, simulationModuleOp, model,
          explicitEquationsMap, idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createGetIDATimeFunction(
          builder, simulationModuleOp, model, idaInstance.get()))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult IDASolver::createInitICSolversFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    const Model<ScheduledEquationsBlock>& model,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(simulationModuleOp.getBody());

  mlir::Location loc = model.getOperation().getLoc();

  llvm::SmallVector<mlir::Type> solverTypes;
  solverTypes.push_back(mlir::ida::InstanceType::get(builder.getContext()));

  mlir::TypeRange variableTypes =
      model.getOperation().getBodyRegion().getArgumentTypes();

  auto initSolversOp =
      builder.create<mlir::simulation::InitICSolversFunctionOp>(
          loc, solverTypes, variableTypes);

  mlir::Block* entryBlock = initSolversOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  return createInitSolversFunction(
      builder,
      initSolversOp.getVariables(),
      model,
      idaInstance);
}

mlir::LogicalResult IDASolver::createDeinitICSolversFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(simulationModuleOp.getBody());

  mlir::Location loc = simulationModuleOp.getLoc();

  llvm::SmallVector<mlir::Type> solverTypes;
  solverTypes.push_back(mlir::ida::InstanceType::get(builder.getContext()));

  auto deinitSolversOp =
      builder.create<mlir::simulation::DeinitICSolversFunctionOp>(
          loc, solverTypes);

  mlir::Block* entryBlock = deinitSolversOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  return createDeinitSolversFunction(
      builder, deinitSolversOp.getSolvers()[0], idaInstance);
}

mlir::LogicalResult IDASolver::createInitMainSolversFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    const Model<ScheduledEquationsBlock>& model,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(simulationModuleOp.getBody());

  mlir::Location loc = model.getOperation().getLoc();

  llvm::SmallVector<mlir::Type> solverTypes;
  solverTypes.push_back(mlir::ida::InstanceType::get(builder.getContext()));

  mlir::TypeRange variableTypes =
      model.getOperation().getBodyRegion().getArgumentTypes();

  auto initSolversOp =
      builder.create<mlir::simulation::InitMainSolversFunctionOp>(
          loc, solverTypes, variableTypes);

  mlir::Block* entryBlock = initSolversOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  return createInitSolversFunction(
      builder,
      initSolversOp.getVariables(),
      model,
      idaInstance);
}

mlir::LogicalResult IDASolver::createDeinitMainSolversFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(simulationModuleOp.getBody());

  mlir::Location loc = simulationModuleOp.getLoc();

  llvm::SmallVector<mlir::Type> solverTypes;
  solverTypes.push_back(mlir::ida::InstanceType::get(builder.getContext()));

  auto deinitSolversOp =
      builder.create<mlir::simulation::DeinitMainSolversFunctionOp>(
          loc, solverTypes);

  mlir::Block* entryBlock = deinitSolversOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  return createDeinitSolversFunction(
      builder, deinitSolversOp.getSolvers()[0], idaInstance);
}

mlir::LogicalResult IDASolver::createInitSolversFunction(
    mlir::OpBuilder& builder,
    mlir::ValueRange variables,
    const Model<ScheduledEquationsBlock>& model,
    IDAInstance* idaInstance) const
{
  mlir::Location loc = model.getOperation().getLoc();

  // Create the IDA instance.
  auto instance = idaInstance->createInstance(builder, loc);

  // Configure the IDA instance.
  if (mlir::failed(idaInstance->configure(
          builder, instance, model, variables))) {
    return mlir::failure();
  }

  // Terminate the function.
  builder.create<mlir::simulation::YieldOp>(loc, instance);

  return mlir::success();
}

mlir::LogicalResult IDASolver::createDeinitSolversFunction(
    mlir::OpBuilder& builder,
    mlir::Value instance,
    IDAInstance* idaInstance) const
{
  mlir::Location loc = instance.getLoc();

  // Deallocate the solver.
  if (mlir::failed(idaInstance->deleteInstance(builder, instance))) {
    return mlir::failure();
  }

  // Create the block terminator.
  builder.create<mlir::simulation::YieldOp>(loc, llvm::None);

  return mlir::success();
}

mlir::LogicalResult IDASolver::createSolveICModelFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    const Model<ScheduledEquationsBlock>& model,
    const ExplicitEquationsMap& explicitEquationsMap,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto modelOp = model.getOperation();
  auto loc = modelOp.getLoc();

  // Create the function inside the parent module.
  builder.setInsertionPointToEnd(simulationModuleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "solveICModel",
      mlir::ida::InstanceType::get(builder.getContext()),
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

  if (mlir::failed(idaInstance->performCalcIC(
          builder, functionOp.getSolvers()[0]))) {
    return mlir::failure();
  }

  // Convert the equations into algorithmic code.
  size_t equationTemplateCounter = 0;
  size_t equationCounter = 0;

  std::map<EquationTemplateInfo, EquationTemplate> equationTemplatesMap;
  llvm::ThreadPool threadPool;

  for (const auto& scheduledBlock : model.getScheduledBlocks()) {
    for (const auto& equation : *scheduledBlock) {
      if (idaInstance->hasEquation(equation.get())) {
        // Let IDA process the equation.
        continue;

      } else {
        // The equation is handled by MARCO.
        auto templateFunction = getOrCreateEquationTemplateFunction(
            threadPool, builder, modelOp, equation.get(), explicitEquationsMap,
            equationTemplatesMap, "initial_eq_template_",
            equationTemplateCounter);

        if (!templateFunction.has_value()) {
          equation->getOperation().emitError(
              "The equation can't be made explicit");

          equation->getOperation().dump();
          return mlir::failure();
        }

        // Create the function that calls the template.
        // This function dictates the indices the template will work with.
        std::string equationFunctionName =
            "initial_eq_" + std::to_string(equationCounter);
        ++equationCounter;

        // Collect the variables to be passed to the instantiated template
        // function.
        std::vector<mlir::Value> usedVariables;
        std::vector<mlir::Type> usedVariablesTypes;

        usedVariables.push_back(allVariables[0]);
        usedVariablesTypes.push_back(allVariables[0].getType());

        for (unsigned int position : templateFunction->usedVariables) {
          usedVariables.push_back(allVariables[position + 1]);
          usedVariablesTypes.push_back(allVariables[position + 1].getType());
        }

        // Create the instantiated template function.
        auto equationFunction = createEquationFunction(
            builder, *equation, equationFunctionName,
            templateFunction->funcOp, usedVariablesTypes);

        // Create the call to the instantiated template function.
        builder.create<mlir::func::CallOp>(
            loc, equationFunction, usedVariables);
      }
    }
  }

  // Terminate the function.
  builder.create<mlir::simulation::ReturnOp>(loc, llvm::None);

  return mlir::success();
}

mlir::LogicalResult IDASolver::createCalcICFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    const Model<ScheduledEquationsBlock>& model,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto modelOp = model.getOperation();
  auto loc = modelOp.getLoc();

  // Create the function inside the parent module.
  builder.setInsertionPointToEnd(simulationModuleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "calcIC",
      mlir::ida::InstanceType::get(builder.getContext()),
      RealType::get(builder.getContext()),
      simulationModuleOp.getVariablesTypes(),
      llvm::None, llvm::None);

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Instruct IDA to compute the initial values.
  if (mlir::failed(idaInstance->performCalcIC(
          builder, functionOp.getSolvers()[0]))) {
    return mlir::failure();
  }

  // Terminate the function.
  builder.create<mlir::simulation::ReturnOp>(loc, llvm::None);

  return mlir::success();
}

mlir::LogicalResult IDASolver::createUpdateIDAVariablesFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    const Model<ScheduledEquationsBlock>& model,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto modelOp = model.getOperation();
  auto loc = modelOp.getLoc();

  // Create the function inside the parent module.
  builder.setInsertionPointToEnd(simulationModuleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "updateIDAVariables",
      mlir::ida::InstanceType::get(builder.getContext()),
      RealType::get(builder.getContext()),
      simulationModuleOp.getVariablesTypes(),
      llvm::None, llvm::None);

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  if (mlir::failed(idaInstance->performStep(
          builder, functionOp.getSolvers()[0]))) {
    return mlir::failure();
  }

  builder.create<mlir::simulation::ReturnOp>(loc, llvm::None);
  return mlir::success();
}

mlir::LogicalResult IDASolver::createUpdateNonIDAVariablesFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    const Model<ScheduledEquationsBlock>& model,
    const ExplicitEquationsMap& explicitEquationsMap,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto modelOp = model.getOperation();
  auto loc = modelOp.getLoc();

  // Create the function inside the parent module.
  builder.setInsertionPointToEnd(simulationModuleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "updateNonIDAVariables",
      mlir::ida::InstanceType::get(builder.getContext()),
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

  for (const auto& scheduledBlock : model.getScheduledBlocks()) {
    for (const auto& equation : *scheduledBlock) {
      if (idaInstance->hasEquation(equation.get())) {
        // Let the external solver process the equation.
        continue;

      } else {
        // The equation is handled by MARCO.
        auto templateFunction = getOrCreateEquationTemplateFunction(
            threadPool, builder, modelOp, equation.get(), explicitEquationsMap,
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

        for (unsigned int position : templateFunction->usedVariables) {
          usedVariables.push_back(allVariables[position + 1]);
          usedVariablesTypes.push_back(allVariables[position + 1].getType());
        }

        auto equationFunction = createEquationFunction(
            builder, *equation, equationFunctionName,
            templateFunction->funcOp, usedVariablesTypes);

        // Create the call to the instantiated template function.
        builder.create<mlir::func::CallOp>(
            loc, equationFunction, usedVariables);
      }
    }
  }

  // Terminate the function.
  builder.create<mlir::simulation::ReturnOp>(loc, llvm::None);

  return mlir::success();
}

mlir::LogicalResult IDASolver::createGetIDATimeFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    const Model<ScheduledEquationsBlock>& model,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto modelOp = model.getOperation();
  auto loc = modelOp.getLoc();

  // Create the function inside the parent module.
  builder.setInsertionPointToEnd(simulationModuleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "getIDATime",
      mlir::ida::InstanceType::get(builder.getContext()),
      RealType::get(builder.getContext()),
      simulationModuleOp.getVariablesTypes(),
      llvm::None,
      RealType::get(builder.getContext()));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  mlir::Value increasedTime = idaInstance->getCurrentTime(
      builder, functionOp.getSolvers()[0],
      RealType::get(builder.getContext()));

  // Terminate the function.
  builder.create<mlir::simulation::ReturnOp>(loc, increasedTime);

  return mlir::success();
}

mlir::func::FuncOp IDASolver::createEquationFunction(
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

  std::vector<mlir::Value> args;
  auto iterationRangesSet = equation.getIterationRanges();
  assert(iterationRangesSet.isSingleMultidimensionalRange());//todo: handle ragged case
  auto iterationRanges = iterationRangesSet.minContainingRange();

  for (size_t i = 0, e = equation.getNumOfIterationVars(); i < e; ++i) {
    auto values = valuesFn(equation.getSchedulingDirection(), iterationRanges[i]);

    args.push_back(std::get<0>(values));
    args.push_back(std::get<1>(values));
    args.push_back(std::get<2>(values));
  }

  mlir::ValueRange vars = function.getArguments();
  args.insert(args.end(), vars.begin(), vars.end());

  // Call the equation template function
  builder.create<mlir::func::CallOp>(loc, templateFunction, args);

  builder.create<mlir::func::ReturnOp>(loc);
  return function;
}

namespace
{
  class IDAPass : public mlir::modelica::impl::IDAPassBase<IDAPass>
  {
    public:
      using IDAPassBase::IDAPassBase;

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

        IDASolver solver(cleverDAE);

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
  std::unique_ptr<mlir::Pass> createIDAPass()
  {
    return std::make_unique<IDAPass>();
  }

  std::unique_ptr<mlir::Pass> createIDAPass(const IDAPassOptions& options)
  {
    return std::make_unique<IDAPass>(options);
  }
}