#include "marco/Codegen/Transforms/IDA.h"
#include "marco/Codegen/Transforms/SolverPassBase.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Dialect/KINSOL/KINSOLDialect.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "marco/Codegen/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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
    llvm::SmallVector<VariableOp> usedVariables;
  };
}

namespace
{
  class IDAInstance
  {
    public:
      IDAInstance(
        llvm::StringRef identifier,
        const DerivativesMap& derivativesMap,
        bool reducedSystem,
        bool reducedDerivatives,
        bool jacobianOneSweep);

      void setStartTime(double time);

      void setEndTime(double time);

      bool hasVariable(VariableOp variable) const;

      void addParametricVariable(VariableOp variable);

      void addStateVariable(VariableOp variable);

      void addDerivativeVariable(VariableOp variable);

      void addAlgebraicVariable(VariableOp variable);

      bool hasEquation(ScheduledEquation* equation) const;

      void addEquation(ScheduledEquation* equation);

      mlir::Value createInstance(
          mlir::OpBuilder& builder,
          mlir::Location loc);

      mlir::LogicalResult configure(
          mlir::OpBuilder& builder,
          mlir::Value idaInstance,
          const Model<ScheduledEquationsBlock>& model,
          mlir::ValueRange variables,
          llvm::DenseMap<VariableOp, size_t>& variablesPos,
          const mlir::SymbolTable& symbolTable);

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
      bool hasParametricVariable(VariableOp variable) const;

      bool hasAlgebraicVariable(VariableOp variable) const;

      bool hasStateVariable(VariableOp variable) const;

      bool hasDerivativeVariable(VariableOp variable) const;

      mlir::LogicalResult addVariablesToIDA(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Value idaInstance,
          mlir::ValueRange variables,
          llvm::DenseMap<VariableOp, size_t>& variablesPos,
          const mlir::SymbolTable& symbolTable);

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
          const Model<ScheduledEquationsBlock>& model,
          const mlir::SymbolTable& symbolTable,
          llvm::DenseMap<VariableOp, size_t>& variablesPos);

      mlir::LogicalResult addVariableAccessesInfoToIDA(
          mlir::OpBuilder& builder,
          mlir::Value idaInstance,
          const mlir::SymbolTable& symbolTable,
          const Equation& equation,
          mlir::Value idaEquation);

      mlir::LogicalResult createResidualFunction(
          mlir::OpBuilder& builder,
          const mlir::SymbolTable& symbolTable,
          const Equation& equation,
          mlir::Value idaEquation,
          llvm::StringRef residualFunctionName);

      llvm::DenseSet<VariableOp> getIndependentVariablesForAD(
          const Equation& equation, const mlir::SymbolTable& symbolTable);

      mlir::LogicalResult createPartialDerTemplateFunction(
          mlir::OpBuilder& builder,
          const Equation& equation,
          llvm::StringRef templateName,
          const mlir::SymbolTable& symbolTable);

      mlir::modelica::FunctionOp createPartialDerTemplateFromEquation(
          mlir::OpBuilder& builder,
          const mlir::SymbolTable& symbolTable,
          const marco::codegen::Equation& equation,
          llvm::StringRef templateName);

      mlir::LogicalResult createJacobianFunction(
          mlir::OpBuilder& builder,
          const mlir::SymbolTable& symbolTable,
          const Equation& equation,
          llvm::StringRef jacobianFunctionName,
          VariableOp independentVariable,
          llvm::StringRef partialDerTemplateName,
          llvm::DenseMap<VariableOp, size_t>& variablesPos);

      std::string getIDAFunctionName(llvm::StringRef name) const;

      std::vector<VariableOp> getIDAFunctionArgs(
          const mlir::SymbolTable& symbolTable) const;

      std::multimap<VariableOp, std::pair<IndexSet, ScheduledEquation*>>
      getWritesMap(const Model<ScheduledEquationsBlock>& model) const;

      mlir::AffineMap getAccessMap(
          mlir::OpBuilder& builder,
          const AccessFunction& accessFunction) const;

    private:
      /// Instance identifier.
      /// It is used to create unique symbols.
      std::string identifier;

      const DerivativesMap* derivativesMap;

      bool reducedSystem;
      bool reducedDerivatives;
      bool jacobianOneSweep;

      llvm::Optional<double> startTime;
      llvm::Optional<double> endTime;

      /// The parametric variables of the model that are managed by IDA.
      /// A parametric variable is a variable that is immutable.
      llvm::SmallVector<VariableOp> parametricVariables;

      /// The algebraic variables of the model that are managed by IDA.
      /// An algebraic variable is a variable that is not a parameter, state or
      /// derivative.
      llvm::SmallVector<VariableOp> algebraicVariables;

      /// The state variables of the model that are managed by IDA.
      /// A state variable is a variable for which there exists a derivative
      /// variable.
      llvm::SmallVector<VariableOp> stateVariables;

      /// The derivative variables of the model that are managed by IDA.
      /// A derivative variable is a variable that is the derivative of another
      /// variable.
      llvm::SmallVector<VariableOp> derivativeVariables;

      /// The SSA values of the IDA variables representing the algebraic ones.
      llvm::SmallVector<mlir::Value> idaAlgebraicVariables;

      /// The SSA values of the IDA variables representing the state ones.
      llvm::SmallVector<mlir::Value> idaStateVariables;

      /// Map used for a faster lookup of the parametric variable position.
      llvm::DenseMap<VariableOp, size_t> parametricVariablesLookup;

      /// Map used for a faster lookup of the algebraic variable position.
      llvm::DenseMap<VariableOp, size_t> algebraicVariablesLookup;

      /// Map used for a faster lookup of the state variable position.
      llvm::DenseMap<VariableOp, size_t> stateVariablesLookup;

      /// Map used for a faster lookup of the derivative variable position.
      llvm::DenseMap<VariableOp, size_t> derivativeVariablesLookup;

      /// The equations managed by IDA.
      llvm::DenseSet<ScheduledEquation*> equations;
  };
}

IDAInstance::IDAInstance(
    llvm::StringRef identifier,
    const DerivativesMap& derivativesMap,
    bool reducedSystem,
    bool reducedDerivatives,
    bool jacobianOneSweep)
    : identifier(identifier.str()),
      derivativesMap(&derivativesMap),
      reducedSystem(reducedSystem),
      reducedDerivatives(reducedDerivatives),
      jacobianOneSweep(jacobianOneSweep),
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

bool IDAInstance::hasVariable(VariableOp variable) const
{
  return hasParametricVariable(variable) ||
      hasAlgebraicVariable(variable) ||
      hasStateVariable(variable) ||
      hasDerivativeVariable(variable);
}

void IDAInstance::addParametricVariable(VariableOp variable)
{
  if (!hasVariable(variable)) {
    parametricVariables.push_back(variable);
    parametricVariablesLookup[variable] = parametricVariables.size() - 1;
  }
}

void IDAInstance::addAlgebraicVariable(VariableOp variable)
{
  if (!hasVariable(variable)) {
    algebraicVariables.push_back(variable);
    algebraicVariablesLookup[variable] = algebraicVariables.size() - 1;
  }
}

void IDAInstance::addStateVariable(VariableOp variable)
{
  if (!hasVariable(variable)) {
    stateVariables.push_back(variable);
    stateVariablesLookup[variable] = stateVariables.size() - 1;
  }
}

void IDAInstance::addDerivativeVariable(VariableOp variable)
{
  if (!hasVariable(variable)) {
    derivativeVariables.push_back(variable);
    derivativeVariablesLookup[variable] = derivativeVariables.size() - 1;
  }
}

bool IDAInstance::hasParametricVariable(VariableOp variable) const
{
  return parametricVariablesLookup.find(variable) !=
      parametricVariablesLookup.end();
}

bool IDAInstance::hasAlgebraicVariable(VariableOp variable) const
{
  return algebraicVariablesLookup.find(variable) !=
      algebraicVariablesLookup.end();
}

bool IDAInstance::hasStateVariable(VariableOp variable) const
{
  return stateVariablesLookup.find(variable) != stateVariablesLookup.end();
}

bool IDAInstance::hasDerivativeVariable(VariableOp variable) const
{
  return derivativeVariablesLookup.find(variable) !=
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
    mlir::ValueRange variables,
    llvm::DenseMap<VariableOp, size_t>& variablesPos,
    const mlir::SymbolTable& symbolTable)
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
          builder, moduleOp, idaInstance, variables, variablesPos, symbolTable))) {
    return mlir::failure();
  }

  // Add the equations to IDA.
  if (mlir::failed(addEquationsToIDA(builder, idaInstance, model, symbolTable, variablesPos))) {
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
    mlir::ValueRange variables,
    llvm::DenseMap<VariableOp, size_t>& variablesPos,
    const mlir::SymbolTable& symbolTable)
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
    std::string getterName = getIDAFunctionName(
        "getter_" + std::to_string(getterFunctionCounter++));

    return createGetterFunction(builder, module, loc, arrayType, getterName);
  };

  auto getOrCreateSetterFn = [&](ArrayType arrayType) {
    std::string setterName = getIDAFunctionName(
        "setter_" + std::to_string(setterFunctionCounter++));

    return createSetterFunction(builder, module, loc, arrayType, setterName);
  };

  // Parametric variables.
  for (VariableOp variable : parametricVariables) {
    builder.create<mlir::ida::AddParametricVariableOp>(
        loc, idaInstance, variables[variablesPos[variable]]);
  }

  // Algebraic variables.
  for (VariableOp variable : algebraicVariables) {
    auto arrayType = variable.getVariableType().toArrayType();

    std::vector<int64_t> dimensions = getDimensionsFn(arrayType);
    auto getter = getOrCreateGetterFn(arrayType);
    auto setter = getOrCreateSetterFn(arrayType);

    mlir::Value idaVariable =
        builder.create<mlir::ida::AddAlgebraicVariableOp>(
            loc,
            idaInstance,
            variables[variablesPos[variable]],
            builder.getI64ArrayAttr(dimensions),
            getter.getSymName(),
            setter.getSymName());

    idaAlgebraicVariables.push_back(idaVariable);
  }

  // State variables.
  for (VariableOp variable : stateVariables) {
    auto arrayType = variable.getVariableType().toArrayType();

    std::vector<int64_t> dimensions = getDimensionsFn(arrayType);
    auto getter = getOrCreateGetterFn(arrayType);
    auto setter = getOrCreateSetterFn(arrayType);

    mlir::Value idaVariable =
        builder.create<mlir::ida::AddStateVariableOp>(
            loc,
            idaInstance,
            variables[variablesPos[variable]],
            builder.getI64ArrayAttr(dimensions),
            getter.getSymName(),
            setter.getSymName());

    idaStateVariables.push_back(idaVariable);
  }

  // Derivative variables.
  for (auto stateVariable : llvm::enumerate(stateVariables)) {
    llvm::StringRef derivativeVarName =
        derivativesMap->getDerivative(stateVariable.value().getSymName());

    auto derivativeVariableOp = symbolTable.lookup<VariableOp>(derivativeVarName);
    auto arrayType = derivativeVariableOp.getVariableType().toArrayType();

    std::vector<int64_t> dimensions = getDimensionsFn(arrayType);
    auto getter = getOrCreateGetterFn(arrayType);
    auto setter = getOrCreateSetterFn(arrayType);

    builder.create<mlir::ida::SetDerivativeOp>(
        loc,
        idaInstance,
        idaStateVariables[stateVariable.index()],
        variables[variablesPos[derivativeVariableOp]],
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
    const Model<ScheduledEquationsBlock>& model,
    const mlir::SymbolTable& symbolTable,
    llvm::DenseMap<VariableOp, size_t>& variablesPos)
{
  mlir::Location loc = model.getOperation().getLoc();

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

      auto writingEquations = llvm::make_range(
          writesMap.equal_range(access.getVariable()->getDefiningOp()));

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
             VariableOp variable = access.getVariable()->getDefiningOp();
             return hasVariable(variable);
           });
         }) && "Some non-IDA variables have not been replaced");

  // The accesses to non-IDA variables have been replaced. Now we can proceed
  // to create the residual and jacobian functions.

  // Counters used to obtain unique names for the functions.
  size_t residualFunctionsCounter = 0;
  size_t jacobianFunctionsCounter = 0;
  size_t partialDerTemplatesCounter = 0;

  llvm::DenseMap<VariableOp, mlir::Value> variablesMapping;

  for (const auto& [variable, idaVariable] : llvm::zip(algebraicVariables, idaAlgebraicVariables)) {
    variablesMapping[variable] = idaVariable;
  }

  for (const auto& [variable, idaVariable] : llvm::zip(stateVariables, idaStateVariables)) {
    variablesMapping[variable] = idaVariable;
  }

  for (const auto& [variable, idaVariable] : llvm::zip(derivativeVariables, idaStateVariables)) {
    variablesMapping[variable] = idaVariable;
  }

  for (const auto& equation : independentEquations) {
    auto iterationRanges = equation->getIterationRanges();

    // Keep track of the accessed variables in order to reduce the amount of
    // generated partial derivatives.
    llvm::DenseSet<VariableOp> accessedVariables;

    for (const auto& access : equation->getAccesses()) {
      accessedVariables.insert(access.getVariable()->getDefiningOp());
    }

    llvm::DenseSet<VariableOp> independentVariables =
        getIndependentVariablesForAD(*equation, symbolTable);

    for(auto ranges : llvm::make_range(iterationRanges.rangesBegin(), iterationRanges.rangesEnd())) {
      std::vector<mlir::Attribute> rangesAttr;

      for (size_t i = 0; i < ranges.rank(); ++i) {
        rangesAttr.push_back(builder.getI64ArrayAttr({ ranges[i].getBegin(), ranges[i].getEnd() }));
      }

      auto writtenVar = equation->getWrite().getVariable()->getDefiningOp();

      auto idaEquation = builder.create<mlir::ida::AddEquationOp>(
          equation->getOperation().getLoc(),
          idaInstance,
          builder.getArrayAttr(rangesAttr),
          variablesMapping[writtenVar],
          getAccessMap(builder, equation->getWrite().getAccessFunction()));

      if (reducedDerivatives) {
        if (mlir::failed(addVariableAccessesInfoToIDA(
                builder, idaInstance, symbolTable, *equation, idaEquation))) {
          return mlir::failure();
        }
      }

      // Create the residual function.
      std::string residualFunctionName = getIDAFunctionName(
          "residualFunction_" + std::to_string(residualFunctionsCounter++));

      if (mlir::failed(createResidualFunction(
              builder, symbolTable, *equation, idaEquation, residualFunctionName))) {
        return mlir::failure();
      }

      builder.create<mlir::ida::SetResidualOp>(
          loc, idaInstance, idaEquation, residualFunctionName);

      // Create the partial derivative template.
      std::string partialDerTemplateName = getIDAFunctionName(
          "pder_" + std::to_string(partialDerTemplatesCounter++));

      if (mlir::failed(createPartialDerTemplateFunction(
              builder, *equation, partialDerTemplateName, symbolTable))) {
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
        if (reducedDerivatives &&
            !accessedVariables.contains(variable)) {
          continue;
        }

        std::string jacobianFunctionName = getIDAFunctionName(
            "jacobianFunction_" + std::to_string(jacobianFunctionsCounter++));

        if (mlir::failed(createJacobianFunction(
                builder, symbolTable, *equation, jacobianFunctionName, variable,
                partialDerTemplateName, variablesPos))) {
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
        if (reducedDerivatives &&
            !accessedVariables.contains(variable) &&
            !accessedVariables.contains(
                symbolTable.lookup<VariableOp>(derivativesMap->getDerivative(variable.getSymName())))) {
          continue;
        }

        std::string jacobianFunctionName = getIDAFunctionName(
            "jacobianFunction_" + std::to_string(jacobianFunctionsCounter++));

        if (mlir::failed(createJacobianFunction(
                builder, symbolTable, *equation, jacobianFunctionName, variable,
                partialDerTemplateName, variablesPos))) {
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
    const mlir::SymbolTable& symbolTable,
    const Equation& equation,
    mlir::Value idaEquation)
{
  assert(idaEquation.getType().isa<mlir::ida::EquationType>());

  mlir::Location loc = equation.getOperation().getLoc();

  auto getIDAVariable = [&](VariableOp variable) -> mlir::Value {
    if (derivativesMap->isDerivative(variable.getSymName())) {
      llvm::StringRef stateVarName =
          derivativesMap->getDerivedVariable(variable.getSymName());

      auto stateVar = symbolTable.lookup<VariableOp>(stateVarName);
      return idaStateVariables[stateVariablesLookup[stateVar]];
    }

    if (derivativesMap->hasDerivative(variable.getSymName())) {
      return idaStateVariables[stateVariablesLookup[variable]];
    }

    return idaAlgebraicVariables[algebraicVariablesLookup[variable]];
  };

  // Keep track of the discovered accesses in order to avoid adding the same
  // access map multiple times for the same variable.
  llvm::DenseMap<mlir::Value, llvm::DenseSet<mlir::AffineMap>> maps;

  for (const Access& access : equation.getAccesses()) {
    VariableOp variable = access.getVariable()->getDefiningOp();

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
    const mlir::SymbolTable& symbolTable,
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
  std::vector<VariableOp> managedVariables = getIDAFunctionArgs(symbolTable);

  llvm::SmallVector<mlir::Type> variableTypes;

  for (VariableOp variableOp : managedVariables) {
    variableTypes.push_back(variableOp.getVariableType().toArrayType());
  }

  auto residualFunction = builder.create<mlir::ida::ResidualFunctionOp>(
      loc,
      residualFunctionName,
      RealType::get(builder.getContext()),
      variableTypes,
      equation.getNumOfIterationVars(),
      RealType::get(builder.getContext()));

  mlir::Block* bodyBlock = residualFunction.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);

  // Map the original variables to the ones received by the function, which
  // are in a possibly different order.
  llvm::StringMap<mlir::Value> variablesMap;

  for (auto variableOp : llvm::enumerate(managedVariables)) {
    mlir::Value mapped = residualFunction.getVariables()[variableOp.index()];
    variablesMap[variableOp.value().getSymName()] = mapped;
  }

  // Map for the SSA values.
  mlir::BlockAndValueMapping mapping;

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
    if (auto variableGetOp = mlir::dyn_cast<VariableGetOp>(op)) {
      mlir::Value mapped = variablesMap[variableGetOp.getVariable()];

      if (auto arrayType = mapped.getType().dyn_cast<ArrayType>();
          arrayType && arrayType.isScalar()) {
        mapped = builder.create<LoadOp>(variableGetOp.getLoc(), mapped, llvm::None);
      }

      mapping.map(variableGetOp, mapped);
    } else if (auto timeOp = mlir::dyn_cast<TimeOp>(op)) {
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

llvm::DenseSet<VariableOp> IDAInstance::getIndependentVariablesForAD(
    const Equation& equation, const mlir::SymbolTable& symbolTable)
{
  llvm::DenseSet<VariableOp> result;

  for (const auto& access : equation.getAccesses()) {
    VariableOp var = access.getVariable()->getDefiningOp();
    result.insert(var);

    if (derivativesMap->hasDerivative(var.getSymName())) {
      auto derVar = symbolTable.lookup<VariableOp>(
          derivativesMap->getDerivative(var.getSymName()));
      result.insert(derVar);
    } else if (derivativesMap->isDerivative(var.getSymName())) {
      auto stateVar = symbolTable.lookup<VariableOp>(
          derivativesMap->getDerivedVariable(var.getSymName()));

      result.insert(stateVar);
    }
  }

  return result;
}

mlir::LogicalResult IDAInstance::createPartialDerTemplateFunction(
    mlir::OpBuilder& builder,
    const Equation& equation,
    llvm::StringRef templateName,
    const mlir::SymbolTable& symbolTable)
{
  mlir::Location loc = equation.getOperation().getLoc();

  auto partialDerTemplate = createPartialDerTemplateFromEquation(
      builder, symbolTable, equation, templateName);

  // Add the time to the input variables (and signature).
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(partialDerTemplate.bodyBlock());

  auto timeVariable = builder.create<VariableOp>(
      loc, "time",
      VariableType::get(
          llvm::None,
          RealType::get(builder.getContext()),
          VariabilityProperty::none,
          IOProperty::input));

  // Replace the TimeOp with the newly created variable.
  llvm::SmallVector<TimeOp> timeOps;

  partialDerTemplate.walk([&](TimeOp timeOp) {
    timeOps.push_back(timeOp);
  });

  for (TimeOp timeOp : timeOps) {
    builder.setInsertionPoint(timeOp);

    mlir::Value time = builder.create<VariableGetOp>(
        timeVariable.getLoc(),
        timeVariable.getVariableType().unwrap(),
        timeVariable.getSymName());

    timeOp.replaceAllUsesWith(time);
    timeOp.erase();
  }

  return mlir::success();
}

FunctionOp IDAInstance::createPartialDerTemplateFromEquation(
    mlir::OpBuilder& builder,
    const mlir::SymbolTable& symbolTable,
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
  std::vector<VariableOp> managedVariables = getIDAFunctionArgs(symbolTable);

  // Create the function to be derived.
  auto functionOp = builder.create<FunctionOp>(loc, functionOpName);

  // Start the body of the function.
  builder.setInsertionPointToStart(functionOp.bodyBlock());

  // Clone the original variables.
  for (VariableOp variableOp : managedVariables) {
    auto clonedVariableOp = mlir::cast<VariableOp>(
        builder.cloneWithoutRegions(*variableOp.getOperation()));

    VariableType variableType =
        clonedVariableOp.getVariableType().withIOProperty(IOProperty::input);

    clonedVariableOp.setType(variableType);
  }

  // Create the induction variables.
  llvm::SmallVector<VariableOp> inductionVariables;

  for (size_t i = 0; i < equation.getNumOfIterationVars(); ++i) {
    std::string variableName = "ind" + std::to_string(i);

    auto variableType = VariableType::wrap(
        builder.getIndexType(),
        VariabilityProperty::none,
        IOProperty::input);

    auto variableOp = builder.create<VariableOp>(
        loc, variableName, variableType);

    inductionVariables.push_back(variableOp);
  }

  // Create the output variable, that is the difference between its equation
  // right-hand side value and its left-hand side value.
  auto originalTerminator = mlir::cast<EquationSidesOp>(
      equation.getOperation().bodyBlock()->getTerminator());

  assert(originalTerminator.getLhsValues().size() == 1);
  assert(originalTerminator.getRhsValues().size() == 1);

  auto outputVariable = builder.create<VariableOp>(
      loc, "out",
      VariableType::wrap(
          RealType::get(builder.getContext()),
          VariabilityProperty::none,
          IOProperty::output));

  // Create the body of the function.
  auto algorithmOp = builder.create<AlgorithmOp>(loc);

  builder.setInsertionPointToStart(
      builder.createBlock(&algorithmOp.getBodyRegion()));

  mlir::BlockAndValueMapping mapping;

  // Get the values of the induction variables.
  llvm::SmallVector<mlir::Value> inductions;

  for (VariableOp variableOp : inductionVariables) {
    mlir::Value induction = builder.create<VariableGetOp>(
        variableOp.getLoc(),
        variableOp.getVariableType().unwrap(),
        variableOp.getSymName());

    inductions.push_back(induction);
  }

  auto explicitEquationInductions = equation.getInductionVariables();

  for (const auto& originalInduction :
       llvm::enumerate(explicitEquationInductions)) {
    assert(originalInduction.index() < inductions.size());

    mapping.map(
        originalInduction.value(),
        inductions[originalInduction.index()]);
  }

  // Clone the original operations and compute the residual value.

  for (auto& op : equation.getOperation().bodyBlock()->getOperations()) {
    if (auto equationSidesOp = mlir::dyn_cast<EquationSidesOp>(op)) {
      assert(equationSidesOp.getLhsValues().size() == 1);
      assert(equationSidesOp.getRhsValues().size() == 1);

      mlir::Value lhs = mapping.lookup(equationSidesOp.getLhsValues()[0]);
      mlir::Value rhs = mapping.lookup(equationSidesOp.getRhsValues()[0]);

      if (auto arrayType = lhs.getType().dyn_cast<ArrayType>()) {
        assert(rhs.getType().isa<ArrayType>());
        assert(arrayType.getRank() + explicitEquationInductions.size() == inductions.size());
        auto implicitInductions = llvm::makeArrayRef(inductions).take_back(arrayType.getRank());

        lhs = builder.create<LoadOp>(loc, lhs, implicitInductions);
        rhs = builder.create<LoadOp>(loc, rhs, implicitInductions);
      }

      auto result = builder.create<SubOp>(
          loc, RealType::get(builder.getContext()), rhs, lhs);

      builder.create<VariableSetOp>(loc, outputVariable.getSymName(), result);
      mapping.lookup(equationSidesOp.getLhs()).getDefiningOp<EquationSideOp>().erase();
      mapping.lookup(equationSidesOp.getRhs()).getDefiningOp<EquationSideOp>().erase();
    } else {
      builder.clone(op, mapping);
    }
  }

  // Create the derivative template function.
  ForwardAD forwardAD;

  auto derTemplate = forwardAD.createPartialDerTemplateFunction(
      builder, loc, functionOp, templateName);

  functionOp.erase();

  return derTemplate;
}

mlir::LogicalResult IDAInstance::createJacobianFunction(
    mlir::OpBuilder& builder,
    const mlir::SymbolTable& symbolTable,
    const Equation& equation,
    llvm::StringRef jacobianFunctionName,
    VariableOp independentVariable,
    llvm::StringRef partialDerTemplateName,
    llvm::DenseMap<VariableOp, size_t>& variablesPos)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = equation.getOperation().getLoc();

  // Add the function to the end of the module.
  auto module = equation.getOperation()->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(module.getBody());

  // Create the function.
  std::vector<VariableOp> managedVariables = getIDAFunctionArgs(symbolTable);

  llvm::SmallVector<mlir::Type> variableTypes;

  for (VariableOp variableOp : managedVariables) {
    variableTypes.push_back(variableOp.getVariableType().toArrayType());
  }

  auto jacobianFunction = builder.create<mlir::ida::JacobianFunctionOp>(
      loc,
      jacobianFunctionName,
      RealType::get(builder.getContext()),
      variableTypes,
      equation.getNumOfIterationVars(),
      independentVariable.getVariableType().getRank(),
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
      [&](VariableOp independentVariableOp,
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

        unsigned int oneSeedPosition = variablesPos[independentVariableOp];
        llvm::Optional<unsigned int> alphaSeedPosition = llvm::None;

        if (jacobianOneSweep && derivativesMap->hasDerivative(independentVariableOp.getSymName())) {
          llvm::StringRef derName = derivativesMap->getDerivative(independentVariableOp.getSymName());
          auto op = symbolTable.lookup<VariableOp>(derName);
          alphaSeedPosition = variablesPos[op];
        }

        // Create the seed values for the variables.
        for (VariableOp var : managedVariables) {
          if (!var.getVariableType().isScalar()) {
            ArrayType arrayType = var.getVariableType().toArrayType();
            assert(arrayType.hasStaticShape());

            auto array = builder.create<ArrayBroadcastOp>(
                loc,
                arrayType.toElementType(RealType::get(builder.getContext())),
                zero);

            seedArrays.push_back(array);
            args.push_back(array);

            if (variablesPos[var] == oneSeedPosition) {
              builder.create<StoreOp>(
                  loc, one, array,
                  jacobianFunction.getVariableIndices());

            } else if (alphaSeedPosition.has_value() && variablesPos[var] == *alphaSeedPosition) {
              builder.create<StoreOp>(
                  loc,
                  jacobianFunction.getAlpha(),
                  array,
                  jacobianFunction.getVariableIndices());
            }
          } else {
            if (variablesPos[var] == oneSeedPosition) {
              args.push_back(one);
            } else if (alphaSeedPosition.has_value() && variablesPos[var] == *alphaSeedPosition) {
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
  populateArgsFn(independentVariable, args, arraySeeds);

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

  if (!jacobianOneSweep && derivativesMap->hasDerivative(independentVariable.getSymName())) {
    args.clear();
    arraySeeds.clear();

    populateArgsFn(
        symbolTable.lookup<VariableOp>(
            derivativesMap->getDerivative(independentVariable.getSymName())),
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

std::string IDAInstance::getIDAFunctionName(llvm::StringRef name) const
{
  return "ida_" + identifier + "_" + name.str();
}

std::vector<VariableOp> IDAInstance::getIDAFunctionArgs(
    const mlir::SymbolTable& symbolTable) const
{
  std::vector<VariableOp> result;

  // Add the parametric variables.
  for (VariableOp variable : parametricVariables) {
    result.push_back(variable);
  }

  // Add the algebraic variables.
  for (VariableOp variable : algebraicVariables) {
    result.push_back(variable);
  }

  // Add the state variables.
  for (VariableOp variable : stateVariables) {
    result.push_back(variable);
  }

  // Add the derivative variables.
  // The derivatives must be in the same order of the respective state
  // variables.

  for (VariableOp stateVariable : stateVariables) {
    auto derivativeVariableOp = symbolTable.lookup<VariableOp>(
        derivativesMap->getDerivative(stateVariable.getSymName()));

    result.push_back(derivativeVariableOp);
  }

  return result;
}

std::multimap<VariableOp, std::pair<IndexSet, ScheduledEquation*>>
IDAInstance::getWritesMap(const Model<ScheduledEquationsBlock>& model) const
{
  std::multimap<VariableOp, std::pair<IndexSet, ScheduledEquation*>> writesMap;

  for (const auto& equationsBlock : model.getScheduledBlocks()) {
    for (const auto& equation : *equationsBlock) {
      const Access& write = equation->getWrite();

      VariableOp writtenVariable = write.getVariable()->getDefiningOp();

      IndexSet writtenIndices =
          write.getAccessFunction().map(equation->getIterationRanges());

      writesMap.emplace(
          writtenVariable,
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

      IDASolver(
          bool reducedSystem,
          bool reducedDerivatives,
          bool jacobianOneSweep);

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
      bool reducedSystem;
      bool reducedDerivatives;
      bool jacobianOneSweep;
  };
}

/// Get or create the template equation function for a scheduled equation.
static llvm::Optional<EquationTemplate> getOrCreateEquationTemplateFunction(
    llvm::ThreadPool& threadPool,
    mlir::OpBuilder& builder,
    ModelOp modelOp,
    const mlir::SymbolTable& symbolTable,
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

IDASolver::IDASolver(
    bool reducedSystem,
    bool reducedDerivatives,
    bool jacobianOneSweep)
    : reducedSystem(reducedSystem),
      reducedDerivatives(reducedDerivatives),
      jacobianOneSweep(jacobianOneSweep)
{
}

mlir::LogicalResult IDASolver::solveICModel(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    const Model<ScheduledEquationsBlock>& model)
{
  DerivativesMap emptyDerivativesMap;

  auto idaInstance = std::make_unique<IDAInstance>(
      "ic", emptyDerivativesMap, reducedSystem, reducedDerivatives, jacobianOneSweep);

  idaInstance->setStartTime(0);
  idaInstance->setEndTime(0);

  std::set<std::unique_ptr<Equation>> explicitEquationsStorage;
  ExplicitEquationsMap explicitEquationsMap;

  if (reducedSystem) {
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

      idaInstance->addAlgebraicVariable(
          implicitEquation->getWrite().getVariable()->getDefiningOp());
    }

    // Add the cyclic equations to the set of equations managed by IDA,
    // together with their written variables.

    for (ScheduledEquation* cyclicEquation : cyclicEquations) {
      idaInstance->addEquation(cyclicEquation);

      idaInstance->addAlgebraicVariable(
          cyclicEquation->getWrite().getVariable()->getDefiningOp());
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
        VariableOp var =
            scheduledEquation->getWrite().getVariable()->getDefiningOp();

        if (idaInstance->hasVariable(var)) {
          idaInstance->addEquation(scheduledEquation.get());
        }
      }
    }

    // Add the used parameters.

    for (const auto& scheduledBlock : model.getScheduledBlocks()) {
      for (const auto& equation : *scheduledBlock) {
        for (const Access& access : equation->getAccesses()) {
          if (auto var = access.getVariable(); var->isReadOnly()) {
            idaInstance->addParametricVariable(var->getDefiningOp());
          }
        }
      }
    }
  } else {
    for (const auto& scheduledBlock : model.getScheduledBlocks()) {
      // Add all the equations and all the written variables.
      for (const auto& equation : *scheduledBlock) {
        idaInstance->addEquation(equation.get());

        idaInstance->addAlgebraicVariable(
            equation->getWrite().getVariable()->getDefiningOp());

        // Also add the used parameters.
        for (const Access& access : equation->getAccesses()) {
          if (auto accessVar = access.getVariable();
              accessVar->isReadOnly()) {
            idaInstance->addParametricVariable(accessVar->getDefiningOp());
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
  mlir::SymbolTable symbolTable(model.getOperation());
  const DerivativesMap& derivativesMap = model.getDerivativesMap();

  auto idaInstance = std::make_unique<IDAInstance>(
      "main", derivativesMap, reducedSystem, reducedDerivatives, jacobianOneSweep);

  std::set<std::unique_ptr<Equation>> explicitEquationsStorage;
  ExplicitEquationsMap explicitEquationsMap;

  if (reducedSystem) {
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

      VariableOp var =
          implicitEquation->getWrite().getVariable()->getDefiningOp();

      if (derivativesMap.isDerivative(var.getSymName())) {
        idaInstance->addDerivativeVariable(var);
      } else if (derivativesMap.hasDerivative(var.getSymName())) {
        idaInstance->addStateVariable(var);
      } else {
        idaInstance->addAlgebraicVariable(var);
      }
    }

    // Add the cyclic equations to the set of equations managed by IDA,
    // together with their written variables.

    for (ScheduledEquation* cyclicEquation : cyclicEquations) {
      idaInstance->addEquation(cyclicEquation);

      VariableOp var = cyclicEquation->getWrite().getVariable()->getDefiningOp();

      if (derivativesMap.isDerivative(var.getSymName())) {
        idaInstance->addDerivativeVariable(var);
      } else if (derivativesMap.hasDerivative(var.getSymName())) {
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
        VariableOp var =
            scheduledEquation->getWrite().getVariable()->getDefiningOp();

        if (derivativesMap.isDerivative(var.getSymName())) {
          idaInstance->addEquation(scheduledEquation.get());

          // State variable.
          auto stateVariableOp = symbolTable.lookup<VariableOp>(
              derivativesMap.getDerivedVariable(var.getSymName()));

          idaInstance->addStateVariable(stateVariableOp);

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
        VariableOp var =
            scheduledEquation->getWrite().getVariable()->getDefiningOp();

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

          if (variable->isReadOnly()) {
            idaInstance->addParametricVariable(variable->getDefiningOp());
          }
        }
      }
    }
  } else {
    for (const auto& scheduledBlock : model.getScheduledBlocks()) {
      // Add all the equations and all the written variables.
      for (const auto& equation : *scheduledBlock) {
        idaInstance->addEquation(equation.get());

        VariableOp var = equation->getWrite().getVariable()->getDefiningOp();

        if (derivativesMap.isDerivative(var.getSymName())) {
          // State variable.
          auto stateVar = symbolTable.lookup<VariableOp>(
              derivativesMap.getDerivedVariable(var.getSymName()));

          idaInstance->addStateVariable(stateVar);

          // Derivative variable.
          idaInstance->addDerivativeVariable(var);
        } else if (derivativesMap.hasDerivative(var.getSymName())) {
          // State variable.
          idaInstance->addStateVariable(var);

          // Derivative variable.
          auto derVar = symbolTable.lookup<VariableOp>(
              derivativesMap.getDerivative(var.getSymName()));

          idaInstance->addDerivativeVariable(derVar);
        } else {
          idaInstance->addAlgebraicVariable(var);
        }

        // Also add the used parameters.
        for (const Access& access : equation->getAccesses()) {
          if (auto accessVar = access.getVariable();
              accessVar->isReadOnly()) {
            idaInstance->addParametricVariable(accessVar->getDefiningOp());
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

  llvm::SmallVector<mlir::Type> variableTypes;

  for (VariableOp variableOp : model.getOperation().getVariables()) {
    variableTypes.push_back(variableOp.getVariableType().toArrayType());
  }

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

  llvm::SmallVector<mlir::Type> variableTypes;

  for (VariableOp variableOp : model.getOperation().getVariables()) {
    variableTypes.push_back(variableOp.getVariableType().toArrayType());
  }

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
  llvm::DenseMap<VariableOp, size_t> variablesPos;

  for (auto variable : llvm::enumerate(model.getOperation().getVariables())) {
    variablesPos[variable.value()] = variable.index();
  }

  mlir::SymbolTable symbolTable(model.getOperation());

  if (mlir::failed(idaInstance->configure(
          builder, instance, model, variables, variablesPos, symbolTable))) {
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
  mlir::SymbolTable symbolTable(modelOp);
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

  llvm::StringMap<size_t> variablesPos;

  for (auto variable : llvm::enumerate(modelOp.getVariables())) {
    variablesPos[variable.value().getSymName()] = variable.index();
  }

  for (const auto& scheduledBlock : model.getScheduledBlocks()) {
    for (const auto& equation : *scheduledBlock) {
      if (idaInstance->hasEquation(equation.get())) {
        // Let IDA process the equation.
        continue;

      } else {
        // The equation is handled by MARCO.
        auto templateFunction = getOrCreateEquationTemplateFunction(
            threadPool, builder, modelOp, symbolTable, equation.get(), explicitEquationsMap,
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

        for (VariableOp variableOp : templateFunction->usedVariables) {
          usedVariables.push_back(allVariables[variablesPos[variableOp.getSymName()] + 1]);
          usedVariablesTypes.push_back(allVariables[variablesPos[variableOp.getSymName()] + 1].getType());
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
  mlir::SymbolTable symbolTable(modelOp);
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

  llvm::StringMap<size_t> variablesPos;

  for (auto variable : llvm::enumerate(modelOp.getVariables())) {
    variablesPos[variable.value().getSymName()] = variable.index();
  }

  for (const auto& scheduledBlock : model.getScheduledBlocks()) {
    for (const auto& equation : *scheduledBlock) {
      if (idaInstance->hasEquation(equation.get())) {
        // Let the external solver process the equation.
        continue;

      } else {
        // The equation is handled by MARCO.
        auto templateFunction = getOrCreateEquationTemplateFunction(
            threadPool, builder, modelOp, symbolTable, equation.get(), explicitEquationsMap,
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

        IDASolver solver(reducedSystem, reducedDerivatives, jacobianOneSweep);

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
