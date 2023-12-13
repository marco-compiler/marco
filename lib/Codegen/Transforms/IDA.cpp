#include "marco/Codegen/Transforms/IDA.h"
#include "marco/Codegen/Transforms/SolverPassBase.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ida"

namespace mlir::modelica
{
#define GEN_PASS_DEF_IDAPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class IDAInstance
  {
    public:
      IDAInstance(
        llvm::StringRef identifier,
        mlir::SymbolTableCollection& symbolTableCollection,
        const DerivativesMap* derivativesMap,
        bool reducedSystem,
        bool reducedDerivatives,
        bool jacobianOneSweep,
        bool debugInformation);

      void setStartTime(double time);

      void setEndTime(double time);

      bool hasVariable(VariableOp variable) const;

      void addStateVariable(VariableOp variable);

      void addDerivativeVariable(VariableOp variable);

      void addAlgebraicVariable(VariableOp variable);

      bool hasEquation(ScheduledEquationInstanceOp equation) const;

      void addEquation(ScheduledEquationInstanceOp equation);

      mlir::LogicalResult declareInstance(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp moduleOp);

      mlir::LogicalResult initialize(
          mlir::OpBuilder& builder,
          mlir::Location loc);

      mlir::LogicalResult configure(
          mlir::IRRewriter& rewriter,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variableOps,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          llvm::ArrayRef<SCCOp> SCCs);

      mlir::LogicalResult performCalcIC(
          mlir::OpBuilder& builder,
          mlir::Location loc);

      mlir::LogicalResult performStep(
          mlir::OpBuilder& builder,
          mlir::Location loc);

      mlir::Value getCurrentTime(
          mlir::OpBuilder& builder,
          mlir::Location loc);

      mlir::LogicalResult deleteInstance(
          mlir::OpBuilder& builder,
          mlir::Location loc);

    private:
      bool hasAlgebraicVariable(VariableOp variable) const;

      bool hasStateVariable(VariableOp variable) const;

      bool hasDerivativeVariable(VariableOp variable) const;

      mlir::LogicalResult addVariablesToIDA(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          llvm::ArrayRef<VariableOp> variableOps,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap);

      mlir::ida::VariableGetterOp createGetterFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          GlobalVariableOp variable,
          llvm::StringRef functionName);

      mlir::ida::VariableSetterOp createSetterFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          GlobalVariableOp variable,
          llvm::StringRef functionName);

      mlir::LogicalResult addEquationsToIDA(
          mlir::IRRewriter& rewriter,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variableOps,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          llvm::ArrayRef<SCCOp> SCCs,
          llvm::DenseMap<
              mlir::AffineMap,
              mlir::ida::AccessFunctionOp>& accessFunctionsMap);

      mlir::LogicalResult addVariableAccessesInfoToIDA(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          ModelOp modelOp,
          ScheduledEquationInstanceOp equationOp,
          mlir::Value idaEquation,
          llvm::DenseMap<
              mlir::AffineMap,
              mlir::ida::AccessFunctionOp>& accessFunctionsMap,
          size_t& accessFunctionsCounter);

      mlir::ida::AccessFunctionOp getOrCreateAccessFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          mlir::AffineMap access,
          llvm::StringRef functionNamePrefix,
          llvm::DenseMap<
              mlir::AffineMap,
              mlir::ida::AccessFunctionOp>& accessFunctionsMap,
          size_t& accessFunctionsCounter);

      mlir::ida::AccessFunctionOp createAccessFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          mlir::AffineMap access,
          llvm::StringRef functionName);

      mlir::LogicalResult createResidualFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ScheduledEquationInstanceOp equationOp,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          mlir::Value idaEquation,
          llvm::StringRef residualFunctionName);

      mlir::LogicalResult getIndependentVariablesForAD(
          llvm::DenseSet<VariableOp>& result,
          ModelOp modelOp,
          ScheduledEquationInstanceOp equationOp);

      mlir::LogicalResult createPartialDerTemplateFunction(
          mlir::IRRewriter& rewriter,
          mlir::ModuleOp moduleOp,
          llvm::ArrayRef<VariableOp> variableOps,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          ScheduledEquationInstanceOp equationOp,
          const llvm::DenseSet<VariableOp>& independentVariables,
          llvm::DenseMap<VariableOp, size_t>& independentVariablesPos,
          llvm::StringRef templateName);

      mlir::modelica::FunctionOp createPartialDerTemplateFromEquation(
          mlir::IRRewriter& rewriter,
          mlir::ModuleOp moduleOp,
          llvm::ArrayRef<VariableOp> variableOps,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          ScheduledEquationInstanceOp equationOp,
          const llvm::DenseSet<VariableOp>& independentVariables,
          llvm::DenseMap<VariableOp, size_t>& independentVariablesPos,
          llvm::StringRef templateName);

      mlir::LogicalResult createJacobianFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          ScheduledEquationInstanceOp equationOp,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          llvm::StringRef jacobianFunctionName,
          const llvm::DenseSet<VariableOp>& independentVariables,
          const llvm::DenseMap<VariableOp, size_t>& independentVariablesPos,
          VariableOp independentVariable,
          llvm::StringRef partialDerTemplateName);

      std::string getIDAFunctionName(llvm::StringRef name) const;

      std::optional<mlir::SymbolRefAttr>
      getDerivative(mlir::SymbolRefAttr variable) const;

      std::optional<mlir::SymbolRefAttr>
      getDerivedVariable(mlir::SymbolRefAttr derivative) const;

      mlir::LogicalResult getWritesMap(
          ModelOp modelOp,
          llvm::ArrayRef<SCCOp> SCCs,
          std::multimap<VariableOp, std::pair<
              IndexSet, ScheduledEquationInstanceOp>>& writesMap) const;

    private:
      /// Instance identifier.
      /// It is used to create unique symbols.
      std::string identifier;

      mlir::SymbolTableCollection* symbolTableCollection;

      const DerivativesMap* derivativesMap;

      bool reducedSystem;
      bool reducedDerivatives;
      bool jacobianOneSweep;
      bool debugInformation;

      std::optional<double> startTime;
      std::optional<double> endTime;

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

      /// Map used for a faster lookup of the algebraic variable position.
      llvm::DenseMap<VariableOp, size_t> algebraicVariablesLookup;

      /// Map used for a faster lookup of the state variable position.
      llvm::DenseMap<VariableOp, size_t> stateVariablesLookup;

      /// Map used for a faster lookup of the derivative variable position.
      llvm::DenseMap<VariableOp, size_t> derivativeVariablesLookup;

      /// The equations managed by IDA.
      llvm::DenseSet<ScheduledEquationInstanceOp> equations;
  };
}

IDAInstance::IDAInstance(
    llvm::StringRef identifier,
    mlir::SymbolTableCollection& symbolTableCollection,
    const DerivativesMap* derivativesMap,
    bool reducedSystem,
    bool reducedDerivatives,
    bool jacobianOneSweep,
    bool debugInformation)
    : identifier(identifier.str()),
      symbolTableCollection(&symbolTableCollection),
      derivativesMap(derivativesMap),
      reducedSystem(reducedSystem),
      reducedDerivatives(reducedDerivatives),
      jacobianOneSweep(jacobianOneSweep),
      debugInformation(debugInformation),
      startTime(std::nullopt),
      endTime(std::nullopt)
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
  assert(variable != nullptr);

  return hasAlgebraicVariable(variable) ||
      hasStateVariable(variable) ||
      hasDerivativeVariable(variable);
}

void IDAInstance::addAlgebraicVariable(VariableOp variable)
{
  assert(variable != nullptr);

  if (!hasVariable(variable)) {
    algebraicVariables.push_back(variable);
    algebraicVariablesLookup[variable] = algebraicVariables.size() - 1;
  }
}

void IDAInstance::addStateVariable(VariableOp variable)
{
  assert(variable != nullptr);

  if (!hasVariable(variable)) {
    stateVariables.push_back(variable);
    stateVariablesLookup[variable] = stateVariables.size() - 1;
  }
}

void IDAInstance::addDerivativeVariable(VariableOp variable)
{
  assert(variable != nullptr);

  if (!hasVariable(variable)) {
    derivativeVariables.push_back(variable);
    derivativeVariablesLookup[variable] = derivativeVariables.size() - 1;
  }
}

bool IDAInstance::hasAlgebraicVariable(VariableOp variable) const
{
  assert(variable != nullptr);

  return algebraicVariablesLookup.find(variable) !=
      algebraicVariablesLookup.end();
}

bool IDAInstance::hasStateVariable(VariableOp variable) const
{
  assert(variable != nullptr);
  return stateVariablesLookup.find(variable) != stateVariablesLookup.end();
}

bool IDAInstance::hasDerivativeVariable(VariableOp variable) const
{
  assert(variable != nullptr);

  return derivativeVariablesLookup.find(variable) !=
      derivativeVariablesLookup.end();
}

bool IDAInstance::hasEquation(ScheduledEquationInstanceOp equation) const
{
  assert(equation != nullptr);
  return llvm::find(equations, equation) != equations.end();
}

void IDAInstance::addEquation(ScheduledEquationInstanceOp equation)
{
  assert(equation != nullptr);
  equations.insert(equation);
}

mlir::LogicalResult IDAInstance::declareInstance(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::ModuleOp moduleOp)
{
  // Create the instance.
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());

  auto instanceOp = builder.create<mlir::ida::InstanceOp>(loc, identifier);

  // Update the symbol table.
  symbolTableCollection->getSymbolTable(moduleOp).insert(instanceOp);

  return mlir::success();
}

mlir::LogicalResult IDAInstance::initialize(
    mlir::OpBuilder& builder,
    mlir::Location loc)
{
  // Initialize the instance.
  builder.create<mlir::ida::InitOp>(loc, identifier);

  return mlir::success();
}

mlir::LogicalResult IDAInstance::deleteInstance(
    mlir::OpBuilder& builder,
    mlir::Location loc)
{
  builder.create<mlir::ida::FreeOp>(loc, identifier);
  return mlir::success();
}

mlir::LogicalResult IDAInstance::configure(
    mlir::IRRewriter& rewriter,
    mlir::Location loc,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    llvm::ArrayRef<SCCOp> SCCs)
{
  llvm::DenseMap<
      mlir::AffineMap,
      mlir::ida::AccessFunctionOp> accessFunctionsMap;

  if (startTime.has_value()) {
    rewriter.create<mlir::ida::SetStartTimeOp>(
        loc,
        rewriter.getStringAttr(identifier),
        rewriter.getF64FloatAttr(*startTime));
  }

  if (endTime.has_value()) {
    rewriter.create<mlir::ida::SetEndTimeOp>(
        loc,
        rewriter.getStringAttr(identifier),
        rewriter.getF64FloatAttr(*endTime));
  }

  // Add the variables to IDA.
  if (mlir::failed(addVariablesToIDA(
          rewriter, loc, moduleOp, variableOps, localToGlobalVariablesMap))) {
    return mlir::failure();
  }

  // Add the equations to IDA.
  if (mlir::failed(addEquationsToIDA(
          rewriter, loc, moduleOp, modelOp, variableOps,
          localToGlobalVariablesMap, SCCs, accessFunctionsMap))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult IDAInstance::addVariablesToIDA(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::ModuleOp moduleOp,
    llvm::ArrayRef<VariableOp> variableOps,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap)
{
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

  auto createGetterFn =
      [&](GlobalVariableOp globalVariableOp) -> mlir::ida::VariableGetterOp {
        std::string getterName = getIDAFunctionName(
            "getter_" + std::to_string(getterFunctionCounter++));

        return createGetterFunction(
            builder, loc, moduleOp, globalVariableOp, getterName);
      };

  auto createSetterFn =
      [&](GlobalVariableOp globalVariableOp) -> mlir::ida::VariableSetterOp {
        std::string setterName = getIDAFunctionName(
            "setter_" + std::to_string(setterFunctionCounter++));

        return createSetterFunction(
            builder, loc, moduleOp, globalVariableOp, setterName);
      };

  // Algebraic variables.
  for (VariableOp variableOp : algebraicVariables) {
    auto globalVariableOp =
        localToGlobalVariablesMap.lookup(variableOp.getSymName());

    assert(globalVariableOp && "Global variable not found");
    auto arrayType = variableOp.getVariableType().toArrayType();

    std::vector<int64_t> dimensions = getDimensionsFn(arrayType);
    auto getter = createGetterFn(globalVariableOp);
    auto setter = createSetterFn(globalVariableOp);

    auto addVariableOp =
        builder.create<mlir::ida::AddAlgebraicVariableOp>(
            loc,
            identifier,
            builder.getI64ArrayAttr(dimensions),
            getter.getSymName(),
            setter.getSymName());

    if (debugInformation) {
      addVariableOp.setNameAttr(variableOp.getSymNameAttr());
    }

    idaAlgebraicVariables.push_back(addVariableOp);
  }

  // State variables.
  for (VariableOp variableOp : stateVariables) {
    auto stateGlobalVariableOp =
        localToGlobalVariablesMap.lookup(variableOp.getSymName());

    assert(stateGlobalVariableOp && "Global variable not found");
    auto arrayType = variableOp.getVariableType().toArrayType();

    std::optional<mlir::SymbolRefAttr> derivativeName = getDerivative(
        mlir::SymbolRefAttr::get(variableOp.getSymNameAttr()));

    if (!derivativeName) {
      return mlir::failure();
    }

    assert(derivativeName->getNestedReferences().empty());
    auto derivativeGlobalVariableOp = localToGlobalVariablesMap.lookup(
        derivativeName->getRootReference().getValue());

    assert(derivativeGlobalVariableOp && "No global variable not found");

    std::vector<int64_t> dimensions = getDimensionsFn(arrayType);
    auto stateGetter = createGetterFn(stateGlobalVariableOp);
    auto stateSetter = createSetterFn(stateGlobalVariableOp);
    auto derivativeGetter = createGetterFn(derivativeGlobalVariableOp);
    auto derivativeSetter = createSetterFn(derivativeGlobalVariableOp);

    auto addVariableOp =
        builder.create<mlir::ida::AddStateVariableOp>(
            loc,
            identifier,
            builder.getI64ArrayAttr(dimensions),
            stateGetter.getSymName(),
            stateSetter.getSymName(),
            derivativeGetter.getSymName(),
            derivativeSetter.getSymName());

    if (debugInformation) {
      addVariableOp.setNameAttr(variableOp.getSymNameAttr());
    }

    idaStateVariables.push_back(addVariableOp);
  }

  return mlir::success();
}

mlir::ida::VariableGetterOp IDAInstance::createGetterFunction(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::ModuleOp moduleOp,
    GlobalVariableOp variable,
    llvm::StringRef functionName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto variableType = variable.getType();
  assert(variableType.isa<ArrayType>());
  auto variableArrayType = variableType.cast<ArrayType>();

  auto getterOp = builder.create<mlir::ida::VariableGetterOp>(
      loc,
      functionName,
      variableArrayType.getRank());

  symbolTableCollection->getSymbolTable(moduleOp).insert(getterOp);

  mlir::Block* entryBlock = getterOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto receivedIndices = getterOp.getVariableIndices().take_front(
      variableArrayType.getRank());

  mlir::Value globalVariable =
      builder.create<GlobalVariableGetOp>(loc, variable);

  mlir::Value result = builder.create<LoadOp>(
      loc, globalVariable, receivedIndices);

  if (auto requestedResultType = getterOp.getFunctionType().getResult(0);
      result.getType() != requestedResultType) {
    result = builder.create<CastOp>(loc, requestedResultType, result);
  }

  builder.create<mlir::ida::ReturnOp>(loc, result);
  return getterOp;
}

mlir::ida::VariableSetterOp IDAInstance::createSetterFunction(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::ModuleOp moduleOp,
    GlobalVariableOp variable,
    llvm::StringRef functionName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto variableType = variable.getType();
  assert(variableType.isa<ArrayType>());
  auto variableArrayType = variableType.cast<ArrayType>();

  auto setterOp = builder.create<mlir::ida::VariableSetterOp>(
      loc,
      functionName,
      variableArrayType.getRank());

  symbolTableCollection->getSymbolTable(moduleOp).insert(setterOp);

  mlir::Block* entryBlock = setterOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto receivedIndices = setterOp.getVariableIndices().take_front(
      variableArrayType.getRank());

  mlir::Value globalVariable =
      builder.create<GlobalVariableGetOp>(loc, variable);

  mlir::Value value = setterOp.getValue();

  if (auto requestedValueType = variableArrayType.getElementType();
      value.getType() != requestedValueType) {
    value = builder.create<CastOp>(
        loc, requestedValueType, setterOp.getValue());
  }

  builder.create<StoreOp>(loc, value, globalVariable, receivedIndices);
  builder.create<mlir::ida::ReturnOp>(loc);
  return setterOp;
}

mlir::LogicalResult IDAInstance::addEquationsToIDA(
    mlir::IRRewriter& rewriter,
    mlir::Location loc,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    llvm::ArrayRef<SCCOp> SCCs,
    llvm::DenseMap<
        mlir::AffineMap,
        mlir::ida::AccessFunctionOp>& accessFunctionsMap)
{
  // Substitute the accesses to non-IDA variables with the equations writing
  // in such variables.
  llvm::SmallVector<ScheduledEquationInstanceOp> independentEquations;

  // First create the writes map, that is the knowledge of which equation
  // writes into a variable and in which indices.
  std::multimap<
      VariableOp, std::pair<IndexSet, ScheduledEquationInstanceOp>> writesMap;

  if (mlir::failed(getWritesMap(modelOp, SCCs, writesMap))) {
    return mlir::failure();
  }

  // The equations we are operating on.
  std::queue<ScheduledEquationInstanceOp> processedEquations;

  for (ScheduledEquationInstanceOp equation : equations) {
    processedEquations.push(equation);
  }

  LLVM_DEBUG(llvm::dbgs() << "Replacing the non-IDA variables\n");

  while (!processedEquations.empty()) {
    ScheduledEquationInstanceOp equationOp = processedEquations.front();

    LLVM_DEBUG({
      llvm::dbgs() << "Current equation\n";
      equationOp.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    IndexSet equationIndices = equationOp.getIterationSpace();

    // Get the accesses of the equation.
    llvm::SmallVector<VariableAccess> accesses;

    if (mlir::failed(equationOp.getAccesses(
            accesses, *symbolTableCollection))) {
      return mlir::failure();
    }

    // Replace the non-IDA variables.
    bool atLeastOneAccessReplaced = false;

    for (const VariableAccess& access : accesses) {
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

      const AccessFunction& accessFunction = access.getAccessFunction();
      std::optional<IndexSet> readVariableIndices = std::nullopt;

      if (!equationIndices.empty()) {
        readVariableIndices = accessFunction.map(equationIndices);
      }

      auto readVariableName = access.getVariable();
      assert(readVariableName.getNestedReferences().empty());

      auto readVariableOp = symbolTableCollection->lookupSymbolIn<VariableOp>(
          modelOp, readVariableName.getRootReference());

      auto writingEquations =
          llvm::make_range(writesMap.equal_range(readVariableOp));

      for (const auto& entry : writingEquations) {
        ScheduledEquationInstanceOp writingEquationOp = entry.second.second;

        if (hasEquation(writingEquationOp)) {
          // Ignore the equation if it is already managed by IDA.
          continue;
        }

        const IndexSet& writtenVariableIndices = entry.second.first;
        bool overlaps = false;

        if (!readVariableIndices && writtenVariableIndices.empty()) {
          // Scalar replacement.
          overlaps = true;
        } else if (readVariableIndices &&
                   readVariableIndices->overlaps(writtenVariableIndices)) {
          // Vectorized replacement.
          overlaps = true;
        }

        if (!overlaps) {
          continue;
        }

        atLeastOneAccessReplaced = true;

        auto explicitWritingEquationOp = writingEquationOp.cloneAndExplicitate(
            rewriter, *symbolTableCollection);

        if (!explicitWritingEquationOp) {
          return mlir::failure();
        }

        auto eraseExplicitWritingEquation = llvm::make_scope_exit([&]() {
          rewriter.eraseOp(explicitWritingEquationOp);
        });

        llvm::SmallVector<ScheduledEquationInstanceOp> newEquations;

        auto writeAccess =
            explicitWritingEquationOp.getMatchedAccess(*symbolTableCollection);

        if (!writeAccess) {
          return mlir::failure();
        }

        std::optional<IndexSet> newEquationsIndices = std::nullopt;

        if (!equationIndices.empty()) {
          newEquationsIndices = equationIndices;

          newEquationsIndices = equationIndices.intersect(
              accessFunction.inverseMap(
                  writtenVariableIndices, equationIndices));
        }

        std::optional<std::reference_wrapper<const IndexSet>>
            optionalNewEquationIndices = std::nullopt;

        if (newEquationsIndices) {
          optionalNewEquationIndices =
              std::reference_wrapper(*newEquationsIndices);
        }

        if (mlir::failed(equationOp.cloneWithReplacedAccess(
                rewriter, optionalNewEquationIndices, access,
                explicitWritingEquationOp.getTemplate(),
                *writeAccess, newEquations))) {
          return mlir::failure();
        }

        for (ScheduledEquationInstanceOp newEquation : newEquations) {
          processedEquations.push(newEquation);
        }
      }
    }

    if (atLeastOneAccessReplaced) {
      rewriter.eraseOp(equationOp);
    } else {
      independentEquations.push_back(equationOp);
    }

    processedEquations.pop();
  }

  // Check that all the non-IDA variables have been replaced.
  assert(([&]() -> bool {
           llvm::SmallVector<VariableAccess> accesses;

           for (auto equationOp : independentEquations) {
              accesses.clear();

              if (mlir::failed(equationOp.getAccesses(
                      accesses, *symbolTableCollection))) {
                return false;
              }

              for (const VariableAccess& access : accesses) {
                auto variable = access.getVariable();
                assert(variable.getNestedReferences().empty());

                auto variableOp =
                    symbolTableCollection->lookupSymbolIn<VariableOp>(
                        modelOp, variable.getRootReference());

                if (!hasVariable(variableOp)) {
                  if (!variableOp.isReadOnly() || !reducedSystem) {
                    return false;
                  }
                }
              }
           }

           return true;
         })() && "Some non-IDA variables have not been replaced");

  // The accesses to non-IDA variables have been replaced. Now we can proceed
  // to create the residual and jacobian functions.

  // Counters used to obtain unique names for the functions.
  size_t accessFunctionsCounter = 0;
  size_t residualFunctionsCounter = 0;
  size_t jacobianFunctionsCounter = 0;
  size_t partialDerTemplatesCounter = 0;

  llvm::DenseMap<VariableOp, mlir::Value> variablesMapping;

  for (const auto& [variable, idaVariable] :
       llvm::zip(algebraicVariables, idaAlgebraicVariables)) {
    variablesMapping[variable] = idaVariable;
  }

  for (const auto& [variable, idaVariable] :
       llvm::zip(stateVariables, idaStateVariables)) {
    variablesMapping[variable] = idaVariable;
  }

  for (const auto& [variable, idaVariable] :
       llvm::zip(derivativeVariables, idaStateVariables)) {
    variablesMapping[variable] = idaVariable;
  }

  for (ScheduledEquationInstanceOp equationOp : independentEquations) {
    // Keep track of the accessed variables in order to reduce the amount of
    // generated partial derivatives.
    llvm::SmallVector<VariableAccess> accesses;
    llvm::DenseSet<VariableOp> accessedVariables;

    if (mlir::failed(equationOp.getAccesses(
            accesses, *symbolTableCollection))) {
      return mlir::failure();
    }

    for (const VariableAccess& access : accesses) {
      auto variableOp =
          symbolTableCollection->lookupSymbolIn<VariableOp>(
              modelOp, access.getVariable());

      accessedVariables.insert(variableOp);
    }

    // Get the indices of the equation.
    IndexSet equationIndices = equationOp.getIterationSpace();

    if (equationIndices.empty()) {
      equationIndices = IndexSet(Point(0));
    }

    // Get the write access.
    auto writeAccess = equationOp.getMatchedAccess(*symbolTableCollection);

    if (!writeAccess) {
      return mlir::failure();
    }

    auto writtenVar = symbolTableCollection->lookupSymbolIn<VariableOp>(
        modelOp, writeAccess->getVariable());

    // Collect the independent variables for automatic differentiation.
    llvm::DenseSet<VariableOp> independentVariables;

    if (mlir::failed(getIndependentVariablesForAD(
            independentVariables, modelOp, equationOp))) {
      return mlir::failure();
    }

    // Create the partial derivative template.
    std::string partialDerTemplateName = getIDAFunctionName(
        "pder_" + std::to_string(partialDerTemplatesCounter++));

    llvm::DenseMap<VariableOp, size_t> independentVariablesPos;

    if (mlir::failed(createPartialDerTemplateFunction(
            rewriter, moduleOp, variableOps, localToGlobalVariablesMap,
            equationOp, independentVariables, independentVariablesPos,
            partialDerTemplateName))) {
      return mlir::failure();
    }

    for (const MultidimensionalRange& range : llvm::make_range(
             equationIndices.rangesBegin(), equationIndices.rangesEnd())) {
      // Add the equation to the IDA instance.
      auto accessFunctionOp = getOrCreateAccessFunction(
          rewriter, equationOp.getLoc(), moduleOp,
          writeAccess->getAccessFunction().getAffineMap(),
          getIDAFunctionName("access"),
          accessFunctionsMap, accessFunctionsCounter);

      if (!accessFunctionOp) {
        return mlir::failure();
      }

      auto idaEquation = rewriter.create<mlir::ida::AddEquationOp>(
          equationOp.getLoc(),
          identifier,
          mlir::ida::MultidimensionalRangeAttr::get(
              rewriter.getContext(), range),
          variablesMapping[writtenVar],
          accessFunctionOp.getSymName());

      if (debugInformation) {
        std::string stringRepresentation;
        llvm::raw_string_ostream stringOstream(stringRepresentation);
        equationOp.printInline(stringOstream);

        idaEquation.setStringRepresentationAttr(
            rewriter.getStringAttr(stringRepresentation));
      }

      if (reducedDerivatives) {
        // Inform IDA about the accesses performed by the equation.
        if (mlir::failed(addVariableAccessesInfoToIDA(
                rewriter, loc, modelOp, equationOp, idaEquation,
                accessFunctionsMap, accessFunctionsCounter))) {
          return mlir::failure();
        }
      }

      // Create the residual function.
      std::string residualFunctionName = getIDAFunctionName(
          "residualFunction_" + std::to_string(residualFunctionsCounter++));

      if (mlir::failed(createResidualFunction(
              rewriter, moduleOp, equationOp, localToGlobalVariablesMap,
              idaEquation, residualFunctionName))) {
        return mlir::failure();
      }

      rewriter.create<mlir::ida::SetResidualOp>(
          loc, identifier, idaEquation, residualFunctionName);

      // Create the Jacobian functions.
      // Notice that Jacobian functions are not created for derivative
      // variables. Those are already handled when encountering the state
      // variable through the 'alpha' parameter set into the derivative seed.

      assert(algebraicVariables.size() == idaAlgebraicVariables.size());

      for (auto [variable, idaVariable] :
           llvm::zip(algebraicVariables, idaAlgebraicVariables)) {
        if (reducedDerivatives &&
            !accessedVariables.contains(variable)) {
          // The partial derivative is always zero.
          continue;
        }

        std::string jacobianFunctionName = getIDAFunctionName(
            "jacobianFunction_" + std::to_string(jacobianFunctionsCounter++));

        if (mlir::failed(createJacobianFunction(
                rewriter, moduleOp, modelOp, equationOp,
                localToGlobalVariablesMap, jacobianFunctionName,
                independentVariables, independentVariablesPos, variable,
                partialDerTemplateName))) {
          return mlir::failure();
        }

        rewriter.create<mlir::ida::AddJacobianOp>(
            loc, identifier, idaEquation, idaVariable, jacobianFunctionName);
      }

      assert(stateVariables.size() == idaStateVariables.size());

      for (auto [variable, idaVariable] :
           llvm::zip(stateVariables, idaStateVariables)) {
        if (reducedDerivatives &&
            !accessedVariables.contains(variable)) {
          auto derivative = getDerivative(
              mlir::SymbolRefAttr::get(variable.getSymNameAttr()));

          if (!derivative) {
            return mlir::failure();
          }

          assert(derivative->getNestedReferences().empty());

          auto derivativeVariableOp =
              symbolTableCollection->lookupSymbolIn<VariableOp>(
                  modelOp, derivative->getRootReference());

          if (!accessedVariables.contains(derivativeVariableOp)) {
            continue;
          }
        }

        std::string jacobianFunctionName = getIDAFunctionName(
            "jacobianFunction_" + std::to_string(jacobianFunctionsCounter++));

        if (mlir::failed(createJacobianFunction(
                rewriter, moduleOp, modelOp, equationOp,
                localToGlobalVariablesMap, jacobianFunctionName,
                independentVariables, independentVariablesPos, variable,
                partialDerTemplateName))) {
          return mlir::failure();
        }

        rewriter.create<mlir::ida::AddJacobianOp>(
            loc, identifier, idaEquation, idaVariable, jacobianFunctionName);
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult IDAInstance::addVariableAccessesInfoToIDA(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    ModelOp modelOp,
    ScheduledEquationInstanceOp equationOp,
    mlir::Value idaEquation,
    llvm::DenseMap<
        mlir::AffineMap,
        mlir::ida::AccessFunctionOp>& accessFunctionsMap,
    size_t& accessFunctionsCounter)
{
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();
  assert(idaEquation.getType().isa<mlir::ida::EquationType>());

  auto getIDAVariable = [&](VariableOp variableOp) -> mlir::Value {
    if (auto stateVariable = getDerivedVariable(mlir::SymbolRefAttr::get(
            variableOp.getSymNameAttr()))) {
      auto stateVariableOp = symbolTableCollection->lookupSymbolIn<VariableOp>(
          modelOp, *stateVariable);

      return idaStateVariables[stateVariablesLookup[stateVariableOp]];
    }

    if (auto derivativeVariable = getDerivative(mlir::SymbolRefAttr::get(
            variableOp.getSymNameAttr()))) {
      auto derivativeVariableOp =
          symbolTableCollection->lookupSymbolIn<VariableOp>(
              modelOp, *derivativeVariable);

      return idaStateVariables[stateVariablesLookup[derivativeVariableOp]];
    }

    return idaAlgebraicVariables[algebraicVariablesLookup[variableOp]];
  };

  // Keep track of the discovered accesses in order to avoid adding the same
  // access map multiple times for the same variable.
  llvm::DenseMap<mlir::Value, llvm::DenseSet<mlir::AffineMap>> maps;

  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed(equationOp.getAccesses(accesses, *symbolTableCollection))) {
    return mlir::failure();
  }

  for (const VariableAccess& access : accesses) {
    auto variableOp = symbolTableCollection->lookupSymbolIn<VariableOp>(
        modelOp, access.getVariable());

    if (!hasVariable(variableOp)) {
      continue;
    }

    mlir::Value idaVariable = getIDAVariable(variableOp);
    assert(idaVariable != nullptr);
    maps[idaVariable].insert(access.getAccessFunction().getAffineMap());
  }

  // Inform IDA about the discovered accesses.
  for (const auto& entry : maps) {
    mlir::Value idaVariable = entry.getFirst();

    for (mlir::AffineMap map : entry.getSecond()) {
      auto accessFunctionOp = getOrCreateAccessFunction(
          builder, loc, moduleOp, map,
          getIDAFunctionName("access"),
          accessFunctionsMap, accessFunctionsCounter);

      if (!accessFunctionOp) {
        return mlir::failure();
      }

      builder.create<mlir::ida::AddVariableAccessOp>(
          loc, identifier, idaEquation, idaVariable,
          accessFunctionOp.getSymName());
    }
  }

  return mlir::success();
}

mlir::ida::AccessFunctionOp IDAInstance::getOrCreateAccessFunction(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::ModuleOp moduleOp,
    mlir::AffineMap access,
    llvm::StringRef functionNamePrefix,
    llvm::DenseMap<
        mlir::AffineMap,
        mlir::ida::AccessFunctionOp>& accessFunctionsMap,
    size_t& accessFunctionsCounter)
{
  auto it = accessFunctionsMap.find(access);

  if (it == accessFunctionsMap.end()) {
    std::string functionName =
        functionNamePrefix.str() + "_" +
        std::to_string(accessFunctionsCounter++);

    auto accessFunctionOp = createAccessFunction(
        builder, loc, moduleOp, access, functionName);

    if (!accessFunctionOp) {
      return nullptr;
    }

    accessFunctionsMap[access] = accessFunctionOp;
    return accessFunctionOp;
  }

  return it->getSecond();
}

mlir::ida::AccessFunctionOp IDAInstance::createAccessFunction(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::ModuleOp moduleOp,
    mlir::AffineMap access,
    llvm::StringRef functionName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  // Normalize the access so that it has at least one dimension and one result.
  llvm::SmallVector<mlir::AffineExpr> expressions;

  for (mlir::AffineExpr expression : access.getResults()) {
    expressions.push_back(expression);
  }

  if (expressions.empty()) {
    expressions.push_back(
        mlir::getAffineConstantExpr(0, builder.getContext()));
  }

  auto extendedAccess = mlir::AffineMap::get(
      std::max(static_cast<unsigned int>(1), access.getNumDims()),
      access.getNumSymbols(),
      expressions, builder.getContext());

  // Create the operation for the access function.
  auto accessFunctionOp = builder.create<mlir::ida::AccessFunctionOp>(
      loc,
      functionName,
      extendedAccess.getNumDims(),
      extendedAccess.getNumResults());

  symbolTableCollection->getSymbolTable(moduleOp).insert(accessFunctionOp);

  mlir::Block* bodyBlock = accessFunctionOp.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);

  // Materialize the access.
  llvm::SmallVector<mlir::Value, 3> results;

  if (mlir::failed(materializeAffineMap(
          builder, loc, extendedAccess,
          accessFunctionOp.getEquationIndices(), results))) {
    return nullptr;
  }

  builder.create<mlir::ida::ReturnOp>(loc, results);
  return accessFunctionOp;
}

mlir::LogicalResult IDAInstance::createResidualFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    ScheduledEquationInstanceOp equationOp,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    mlir::Value idaEquation,
    llvm::StringRef residualFunctionName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Location loc = equationOp.getLoc();

  size_t numOfInductionVariables = equationOp.getInductionVariables().size() +
      equationOp.getNumOfImplicitInductionVariables();

  auto residualFunction = builder.create<mlir::ida::ResidualFunctionOp>(
      loc,
      residualFunctionName,
      numOfInductionVariables);

  symbolTableCollection->getSymbolTable(moduleOp).insert(residualFunction);

  mlir::Block* bodyBlock = residualFunction.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);

  // Map for the SSA values.
  mlir::IRMapping mapping;

  // Map the iteration variables.
  auto originalInductions = equationOp.getInductionVariables();
  auto mappedInductions = residualFunction.getEquationIndices();
  assert(originalInductions.size() <= mappedInductions.size());

  llvm::SmallVector<mlir::Value, 3> implicitInductions(
      std::next(mappedInductions.begin(), originalInductions.size()),
      mappedInductions.end());

  for (size_t i = 0, e = originalInductions.size(); i < e; ++i) {
    mapping.map(originalInductions[i], mappedInductions[i]);
  }

  for (auto& op : equationOp.getTemplate().getOps()) {
    if (auto timeOp = mlir::dyn_cast<TimeOp>(op)) {
      mlir::Value timeReplacement = residualFunction.getTime();

      timeReplacement = builder.create<CastOp>(
          timeReplacement.getLoc(),
          RealType::get(builder.getContext()),
          timeReplacement);

      mapping.map(timeOp.getResult(), timeReplacement);
    } else if (mlir::isa<EquationSideOp>(op)) {
      continue;
    } else if (auto equationSidesOp = mlir::dyn_cast<EquationSidesOp>(op)) {
      // Compute the difference between the right-hand side and the left-hand
      // side of the equation.
      uint64_t viewElementIndex = equationOp.getViewElementIndex();

      mlir::Value lhs = mapping.lookup(
          equationSidesOp.getLhsValues()[viewElementIndex]);

      mlir::Value rhs = mapping.lookup(
          equationSidesOp.getRhsValues()[viewElementIndex]);

      if (lhs.getType().isa<ArrayType>()) {
        assert(lhs.getType().cast<ArrayType>().getRank() ==
               static_cast<int64_t>(implicitInductions.size()));

        lhs = builder.create<LoadOp>(lhs.getLoc(), lhs, implicitInductions);
      }

      if (rhs.getType().isa<ArrayType>()) {
        assert(rhs.getType().cast<ArrayType>().getRank() ==
               static_cast<int64_t>(implicitInductions.size()));

        rhs = builder.create<LoadOp>(rhs.getLoc(), rhs, implicitInductions);
      }

      mlir::Value difference = builder.create<SubOp>(
          loc, builder.getF64Type(), rhs, lhs);

      builder.create<mlir::ida::ReturnOp>(difference.getLoc(), difference);
    } else if (auto variableGetOp = mlir::dyn_cast<VariableGetOp>(op)) {
      // Replace the local variables with the global ones.
      auto globalVariableOp =
          localToGlobalVariablesMap.lookup(variableGetOp.getVariable());

      mlir::Value globalVariable = builder.create<GlobalVariableGetOp>(
          variableGetOp.getLoc(), globalVariableOp);

      if (globalVariable.getType().cast<ArrayType>().isScalar()) {
        globalVariable = builder.create<LoadOp>(
            globalVariable.getLoc(), globalVariable, std::nullopt);
      }

      mapping.map(variableGetOp.getResult(), globalVariable);
    } else {
      builder.clone(op, mapping);
    }
  }

  return mlir::success();
}

mlir::LogicalResult IDAInstance::getIndependentVariablesForAD(
    llvm::DenseSet<VariableOp>& result,
    ModelOp modelOp,
    ScheduledEquationInstanceOp equationOp)
{
  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed(equationOp.getAccesses(accesses, *symbolTableCollection))) {
    return mlir::failure();
  }

  for (const VariableAccess& access : accesses) {
    auto variableOp =
        symbolTableCollection->lookupSymbolIn<VariableOp>(
            modelOp, access.getVariable());

    result.insert(variableOp);

    if (auto derivative = getDerivative(access.getVariable())) {
      auto derivativeVariableOp =
          symbolTableCollection->lookupSymbolIn<VariableOp>(
              modelOp, *derivative);

      result.insert(derivativeVariableOp);
    }

    if (auto state = getDerivedVariable(access.getVariable())) {
      auto stateVariableOp =
          symbolTableCollection->lookupSymbolIn<VariableOp>(modelOp, *state);

      result.insert(stateVariableOp);
    }
  }

  return mlir::success();
}

mlir::LogicalResult IDAInstance::createPartialDerTemplateFunction(
    mlir::IRRewriter& rewriter,
    mlir::ModuleOp moduleOp,
    llvm::ArrayRef<VariableOp> variableOps,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    ScheduledEquationInstanceOp equationOp,
    const llvm::DenseSet<VariableOp>& independentVariables,
    llvm::DenseMap<VariableOp, size_t>& independentVariablesPos,
    llvm::StringRef templateName)
{
  mlir::Location loc = equationOp.getLoc();

  auto partialDerTemplate = createPartialDerTemplateFromEquation(
      rewriter, moduleOp, variableOps, localToGlobalVariablesMap, equationOp,
      independentVariables, independentVariablesPos, templateName);

  // Add the time to the input variables (and signature).
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(partialDerTemplate.bodyBlock());

  auto timeVariable = rewriter.create<VariableOp>(
      loc, "time",
      VariableType::get(
          std::nullopt,
          RealType::get(rewriter.getContext()),
          VariabilityProperty::none,
          IOProperty::input));

  // Replace the TimeOp with the newly created variable.
  llvm::SmallVector<TimeOp> timeOps;

  partialDerTemplate.walk([&](TimeOp timeOp) {
    timeOps.push_back(timeOp);
  });

  for (TimeOp timeOp : timeOps) {
    rewriter.setInsertionPoint(timeOp);

    mlir::Value time = rewriter.create<VariableGetOp>(
        timeVariable.getLoc(),
        timeVariable.getVariableType().unwrap(),
        timeVariable.getSymName());

    rewriter.replaceOp(timeOp, time);
  }

  return mlir::success();
}

FunctionOp IDAInstance::createPartialDerTemplateFromEquation(
    mlir::IRRewriter& rewriter,
    mlir::ModuleOp moduleOp,
    llvm::ArrayRef<VariableOp> variableOps,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    ScheduledEquationInstanceOp equationOp,
    const llvm::DenseSet<VariableOp>& independentVariables,
    llvm::DenseMap<VariableOp, size_t>& independentVariablesPos,
    llvm::StringRef templateName)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Location loc = equationOp.getLoc();

  // Create the function.
  std::string functionOpName = templateName.str() + "_base";

  // Create the function to be derived.
  auto functionOp = rewriter.create<FunctionOp>(loc, functionOpName);

  // Start the body of the function.
  rewriter.setInsertionPointToStart(functionOp.bodyBlock());

  // Replicate the original independent variables inside the function.
  llvm::StringMap<VariableOp> localVariableOps;
  size_t independentVariableIndex = 0;

  for (VariableOp variableOp : variableOps) {
    if (!independentVariables.contains(variableOp)) {
      continue;
    }

    VariableType variableType =
        variableOp.getVariableType().withIOProperty(IOProperty::input);

    auto clonedVariableOp = rewriter.create<VariableOp>(
        variableOp.getLoc(), variableOp.getSymName(), variableType);

    localVariableOps[variableOp.getSymName()] = clonedVariableOp;
    independentVariablesPos[variableOp] = independentVariableIndex++;
  }

  // Create the induction variables.
  llvm::SmallVector<VariableOp, 3> inductionVariablesOps;

  size_t numOfExplicitInductions = equationOp.getInductionVariables().size();

  size_t numOfImplicitInductions =
      static_cast<size_t>(equationOp.getNumOfImplicitInductionVariables());

  size_t totalAmountOfInductions =
      numOfExplicitInductions + numOfImplicitInductions;

  for (size_t i = 0; i < totalAmountOfInductions; ++i) {
    std::string variableName = "ind" + std::to_string(i);

    auto variableType = VariableType::wrap(
        rewriter.getIndexType(),
        VariabilityProperty::none,
        IOProperty::input);

    auto variableOp = rewriter.create<VariableOp>(
        loc, variableName, variableType);

    inductionVariablesOps.push_back(variableOp);
  }

  // Create the output variable, that is the difference between its equation
  // right-hand side value and its left-hand side value.
  std::string outVariableName = "out";
  size_t outVariableNameCounter = 0;

  while (localToGlobalVariablesMap.count(outVariableName) != 0) {
    outVariableName = "out_" + std::to_string(outVariableNameCounter++);
  }

  auto outputVariableOp = rewriter.create<VariableOp>(
      loc, outVariableName,
      VariableType::wrap(
          RealType::get(rewriter.getContext()),
          VariabilityProperty::none,
          IOProperty::output));

  // Create the body of the function.
  auto algorithmOp = rewriter.create<AlgorithmOp>(loc);

  rewriter.setInsertionPointToStart(
      rewriter.createBlock(&algorithmOp.getBodyRegion()));

  mlir::IRMapping mapping;

  // Get the values of the induction variables.
  auto originalInductions = equationOp.getInductionVariables();
  assert(originalInductions.size() <= inductionVariablesOps.size());

  for (size_t i = 0, e = originalInductions.size(); i < e; ++i) {
    mlir::Value mappedInduction = rewriter.create<VariableGetOp>(
        inductionVariablesOps[i].getLoc(),
        inductionVariablesOps[i].getVariableType().unwrap(),
        inductionVariablesOps[i].getSymName());

    mapping.map(originalInductions[i], mappedInduction);
  }

  llvm::SmallVector<mlir::Value, 3> implicitInductions;

  for (size_t i = 0; i < numOfImplicitInductions; ++i) {
    mlir::Value mappedInduction = rewriter.create<VariableGetOp>(
        inductionVariablesOps[numOfExplicitInductions + i].getLoc(),
        inductionVariablesOps[numOfExplicitInductions + i].getVariableType().unwrap(),
        inductionVariablesOps[numOfExplicitInductions + i].getSymName());

    implicitInductions.push_back(mappedInduction);
  }

  // Determine the operations to be cloned by starting from the terminator and
  // walking through the dependencies.
  llvm::DenseSet<mlir::Operation*> toBeCloned;
  std::stack<mlir::Operation*> toBeClonedVisitStack;

  auto equationSidesOp = mlir::cast<EquationSidesOp>(
      equationOp.getTemplate().getBody()->getTerminator());

  uint64_t viewElementIndex = equationOp.getViewElementIndex();

  mlir::Value lhs = equationSidesOp.getLhsValues()[viewElementIndex];
  mlir::Value rhs = equationSidesOp.getRhsValues()[viewElementIndex];

  if (mlir::Operation* lhsOp = lhs.getDefiningOp()) {
    toBeClonedVisitStack.push(lhsOp);
  }

  if (mlir::Operation* rhsOp = rhs.getDefiningOp()) {
    toBeClonedVisitStack.push(rhsOp);
  }

  while (!toBeClonedVisitStack.empty()) {
    mlir::Operation* op = toBeClonedVisitStack.top();
    toBeClonedVisitStack.pop();

    toBeCloned.insert(op);

    for (mlir::Value operand : op->getOperands()) {
      if (auto operandOp = operand.getDefiningOp()) {
        toBeClonedVisitStack.push(operandOp);
      }
    }
  }

  // Clone the original operations and compute the residual value.
  for (auto& op : equationOp.getTemplate().getOps()) {
    if (!toBeCloned.contains(&op)) {
      continue;
    }

    if (auto globalGetOp = mlir::dyn_cast<GlobalVariableGetOp>(op)) {
      VariableOp variableOp = localVariableOps[globalGetOp.getVariable()];

      auto getOp = rewriter.create<VariableGetOp>(
          globalGetOp.getLoc(), variableOp);

      mapping.map(globalGetOp.getResult(), getOp.getResult());
    } else if (mlir::isa<EquationSideOp, EquationSidesOp>(op)) {
      continue;
    } else {
      rewriter.clone(op, mapping);
    }
  }

  mlir::Value mappedLhs = mapping.lookup(lhs);
  mlir::Value mappedRhs = mapping.lookup(rhs);

  if (mappedLhs.getType().isa<ArrayType>()) {
    assert(mappedLhs.getType().cast<ArrayType>().getRank() ==
           static_cast<int64_t>(implicitInductions.size()));

    mappedLhs = rewriter.create<LoadOp>(
        mappedLhs.getLoc(), mappedLhs, implicitInductions);
  }

  if (mappedRhs.getType().isa<ArrayType>()) {
    assert(mappedRhs.getType().cast<ArrayType>().getRank() ==
           static_cast<int64_t>(implicitInductions.size()));

    mappedRhs = rewriter.create<LoadOp>(
        mappedRhs.getLoc(), mappedRhs, implicitInductions);
  }

  auto result = rewriter.create<SubOp>(
      loc, RealType::get(rewriter.getContext()), mappedRhs, mappedLhs);

  rewriter.create<VariableSetOp>(
      loc, outputVariableOp.getSymName(), result);

  // Create the derivative template function.
  ForwardAD forwardAD;

  auto derTemplate = forwardAD.createPartialDerTemplateFunction(
      rewriter, loc, *symbolTableCollection, functionOp, templateName);

  rewriter.eraseOp(functionOp);

  // Replace the local variables with the global ones.
  llvm::DenseSet<VariableOp> variablesToBeReplaced;

  for (VariableOp variableOp : derTemplate.getVariables()) {
    if (localVariableOps.count(variableOp.getSymName()) != 0) {
      variablesToBeReplaced.insert(variableOp);
    }
  }

  llvm::SmallVector<VariableGetOp> variableGetOps;

  derTemplate.walk([&](VariableGetOp getOp) {
    if (localVariableOps.count(getOp.getVariable()) != 0) {
      variableGetOps.push_back(getOp);
    }
  });

  for (VariableGetOp getOp : variableGetOps) {
    rewriter.setInsertionPoint(getOp);
    assert(localToGlobalVariablesMap.count(getOp.getVariable()) != 0);

    mlir::Value globalVariable = rewriter.create<GlobalVariableGetOp>(
        getOp.getLoc(), localToGlobalVariablesMap.lookup(getOp.getVariable()));

    if (globalVariable.getType().cast<ArrayType>().isScalar()) {
      globalVariable = rewriter.create<LoadOp>(
          globalVariable.getLoc(), globalVariable, std::nullopt);
    }

    rewriter.replaceOp(getOp, globalVariable);
  }

  for (VariableOp variableOp : variablesToBeReplaced) {
    rewriter.eraseOp(variableOp);
  }

  return derTemplate;
}

static GlobalVariableOp createGlobalSeed(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    llvm::StringRef name,
    mlir::Type type)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());

  mlir::Attribute initialValue = nullptr;
  auto arrayType = type.cast<ArrayType>();
  mlir::Type elementType = arrayType.getElementType();

  if (elementType.isa<BooleanType>()) {
    llvm::SmallVector<bool> values(arrayType.getNumElements(), false);
    initialValue = BooleanArrayAttr::get(arrayType, values);
  } else if (elementType.isa<IntegerType>()) {
    llvm::SmallVector<int64_t> values(arrayType.getNumElements(), 0);
    initialValue = IntegerArrayAttr::get(arrayType, values);
  } else if (elementType.isa<RealType>()) {
    llvm::SmallVector<double> values(arrayType.getNumElements(), 0);
    initialValue = RealArrayAttr::get(arrayType, values);
  }

  return builder.create<GlobalVariableOp>(loc, name, type, initialValue);
}

static void setSeed(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    GlobalVariableOp seedVariableOp,
    mlir::ValueRange indices,
    mlir::Value value)
{
  mlir::Value seed = builder.create<GlobalVariableGetOp>(loc, seedVariableOp);
  builder.create<StoreOp>(loc, value, seed, indices);
}

mlir::LogicalResult IDAInstance::createJacobianFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    ScheduledEquationInstanceOp equationOp,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    llvm::StringRef jacobianFunctionName,
    const llvm::DenseSet<VariableOp>& independentVariables,
    const llvm::DenseMap<VariableOp, size_t>& independentVariablesPos,
    VariableOp independentVariable,
    llvm::StringRef partialDerTemplateName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Location loc = equationOp.getLoc();

  size_t numOfIndependentVars = independentVariables.size();

  size_t numOfInductions = equationOp.getInductionVariables().size() +
      equationOp.getNumOfImplicitInductionVariables();

  // Create the function.
  auto jacobianFunction = builder.create<mlir::ida::JacobianFunctionOp>(
      loc, jacobianFunctionName, numOfInductions,
      independentVariable.getVariableType().getRank());

  symbolTableCollection->getSymbolTable(moduleOp).insert(jacobianFunction);

  mlir::Block* bodyBlock = jacobianFunction.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);

  // Create the global seeds for the variables.
  llvm::SmallVector<GlobalVariableOp> varSeeds(numOfIndependentVars, nullptr);
  size_t seedsCounter = 0;

  for (VariableOp variableOp : independentVariables) {
    assert(independentVariablesPos.count(variableOp) != 0);
    size_t pos = independentVariablesPos.lookup(variableOp);

    std::string seedName = jacobianFunctionName.str() + "_seed_" +
        std::to_string(seedsCounter++);

    assert(varSeeds[pos] == nullptr && "Seed already created");

    auto seed = createGlobalSeed(
        builder, moduleOp, loc, seedName,
        variableOp.getVariableType().toArrayType());

    if (!seed) {
      return mlir::failure();
    }

    varSeeds[pos] = seed;
  }

  assert(llvm::none_of(varSeeds, [](GlobalVariableOp seed) {
    return seed == nullptr;
  }) && "Some seeds have not been created");

  // Zero and one constants to be used to update the array seeds or for the
  // scalar seeds.
  mlir::Value zero = builder.create<ConstantOp>(
      loc, RealAttr::get(builder.getContext(), 0));

  mlir::Value one = builder.create<ConstantOp>(
      loc, RealAttr::get(builder.getContext(), 1));

  // Function to collect the arguments to be passed to the derivative template
  // function.
  auto collectArgsFn = [&](llvm::SmallVectorImpl<mlir::Value>& args) {
    // 'Time' variable.
    args.push_back(jacobianFunction.getTime());

    // Equation indices.
    for (mlir::Value equationIndex : jacobianFunction.getEquationIndices()) {
      args.push_back(equationIndex);
    }

    // Seeds of the variables.
    for (GlobalVariableOp globalSeed : varSeeds) {
      mlir::Value seed = builder.create<GlobalVariableGetOp>(loc, globalSeed);

      if (seed.getType().cast<ArrayType>().isScalar()) {
        seed = builder.create<LoadOp>(loc, seed, std::nullopt);
      }

      args.push_back(seed);
    }

    // Seeds of the equation indices. They are all equal to zero.
    for (size_t i = 0; i < jacobianFunction.getEquationIndices().size(); ++i) {
      args.push_back(zero);
    }
  };

  if (jacobianOneSweep) {
    // Perform just one call to the template function.
    assert(independentVariablesPos.count(independentVariable) != 0);

    // Set the seed of the variable to one.
    size_t oneSeedPosition =
        independentVariablesPos.lookup(independentVariable);

    setSeed(builder, loc, varSeeds[oneSeedPosition],
            jacobianFunction.getVariableIndices(), one);

    // Set the seed of the derivative to alpha.
    std::optional<size_t> alphaSeedPosition = std::nullopt;

    if (auto derivative = getDerivative(mlir::SymbolRefAttr::get(
            independentVariable.getSymNameAttr()))) {
      auto derVariableOp =
          symbolTableCollection->lookupSymbolIn<VariableOp>(modelOp, *derivative);

      assert(independentVariablesPos.count(derVariableOp) != 0);
      alphaSeedPosition = independentVariablesPos.lookup(derVariableOp);
    }

    if (alphaSeedPosition) {
      mlir::Value alpha = jacobianFunction.getAlpha();

      alpha = builder.create<CastOp>(
          alpha.getLoc(), RealType::get(builder.getContext()), alpha);

      setSeed(builder, loc, varSeeds[*alphaSeedPosition],
              jacobianFunction.getVariableIndices(), alpha);
    }

    // Call the template function.
    llvm::SmallVector<mlir::Value> args;
    collectArgsFn(args);

    auto templateCall = builder.create<CallOp>(
        loc,
        mlir::SymbolRefAttr::get(builder.getContext(), partialDerTemplateName),
        RealType::get(builder.getContext()),
        args);

    mlir::Value result = templateCall.getResult(0);

    // Reset the seeds.
    setSeed(builder, loc, varSeeds[oneSeedPosition],
            jacobianFunction.getVariableIndices(), zero);

    if (alphaSeedPosition) {
      setSeed(builder, loc, varSeeds[*alphaSeedPosition],
              jacobianFunction.getVariableIndices(), zero);
    }

    // Return the result.
    result = builder.create<CastOp>(loc, builder.getF64Type(), result);
    builder.create<mlir::ida::ReturnOp>(loc, result);
  } else {
    llvm::SmallVector<mlir::Value> args;

    // Perform the first call to the template function.
    assert(independentVariablesPos.count(independentVariable) != 0);

    // Set the seed of the variable to one.
    size_t oneSeedPosition =
        independentVariablesPos.lookup(independentVariable);

    setSeed(builder, loc, varSeeds[oneSeedPosition],
            jacobianFunction.getVariableIndices(), one);

    // Call the template function.
    args.clear();
    collectArgsFn(args);

    auto firstTemplateCall = builder.create<CallOp>(
        loc,
        mlir::SymbolRefAttr::get(builder.getContext(), partialDerTemplateName),
        RealType::get(builder.getContext()),
        args);

    mlir::Value result = firstTemplateCall.getResult(0);

    // Reset the seed of the variable.
    setSeed(builder, loc, varSeeds[oneSeedPosition],
            jacobianFunction.getVariableIndices(), zero);

    if (auto derivative = getDerivative(mlir::SymbolRefAttr::get(
            independentVariable.getSymNameAttr()))) {
      auto globalDerivativeOp =
          symbolTableCollection->lookupSymbolIn<VariableOp>(moduleOp, *derivative);

      assert(independentVariablesPos.count(globalDerivativeOp) != 0);

      size_t derSeedPosition =
          independentVariablesPos.lookup(globalDerivativeOp);

      // Set the seed of the derivative to one.
      setSeed(builder, loc, varSeeds[derSeedPosition],
              jacobianFunction.getVariableIndices(), one);

      // Call the template function.
      args.clear();
      collectArgsFn(args);

      auto secondTemplateCall = builder.create<CallOp>(
          loc,
          mlir::SymbolRefAttr::get(builder.getContext(), partialDerTemplateName),
          RealType::get(builder.getContext()),
          args);

      mlir::Value secondResult = secondTemplateCall.getResult(0);

      mlir::Value secondResultTimesAlpha = builder.create<MulOp>(
          loc, RealType::get(builder.getContext()),
          jacobianFunction.getAlpha(), secondResult);

      result = builder.create<AddOp>(
          loc, RealType::get(builder.getContext()),
          result, secondResultTimesAlpha);
    }

    // Return the result.
    result = builder.create<CastOp>(loc, builder.getF64Type(), result);
    builder.create<mlir::ida::ReturnOp>(loc, result);
  }

  return mlir::success();
}

mlir::LogicalResult IDAInstance::performCalcIC(
    mlir::OpBuilder& builder,
    mlir::Location loc)
{
  builder.create<mlir::ida::CalcICOp>(loc, identifier);
  return mlir::success();
}

mlir::LogicalResult IDAInstance::performStep(
    mlir::OpBuilder& builder,
    mlir::Location loc)
{
  builder.create<mlir::ida::StepOp>(loc, identifier);
  return mlir::success();
}

mlir::Value IDAInstance::getCurrentTime(
    mlir::OpBuilder& builder,
    mlir::Location loc)
{
  return builder.create<mlir::ida::GetCurrentTimeOp>(loc, identifier);
}

std::string IDAInstance::getIDAFunctionName(llvm::StringRef name) const
{
  return identifier + "_" + name.str();
}

std::optional<mlir::SymbolRefAttr>
IDAInstance::getDerivative(mlir::SymbolRefAttr variable) const
{
  if (!derivativesMap) {
    return std::nullopt;
  }

  return derivativesMap->getDerivative(variable);
}

std::optional<mlir::SymbolRefAttr>
IDAInstance::getDerivedVariable(mlir::SymbolRefAttr derivative) const
{
  if (!derivativesMap) {
    return std::nullopt;
  }

  return derivativesMap->getDerivedVariable(derivative);
}

mlir::LogicalResult IDAInstance::getWritesMap(
    ModelOp modelOp,
    llvm::ArrayRef<SCCOp> SCCs,
    std::multimap<
        VariableOp,
        std::pair<IndexSet, ScheduledEquationInstanceOp>>& writesMap) const
{
  for (SCCOp scc : SCCs) {
    for (ScheduledEquationInstanceOp equationOp :
         scc.getOps<ScheduledEquationInstanceOp>()) {
      std::optional<VariableAccess> writeAccess =
          equationOp.getMatchedAccess(*symbolTableCollection);

      if (!writeAccess) {
        return mlir::failure();
      }

      auto writtenVariableOp =
          symbolTableCollection->lookupSymbolIn<VariableOp>(
              modelOp, writeAccess->getVariable());

      const AccessFunction& accessFunction = writeAccess->getAccessFunction();

      if (auto equationIndices = equationOp.getIterationSpace();
          !equationIndices.empty()) {
        IndexSet variableIndices = accessFunction.map(equationIndices);

        writesMap.emplace(
            writtenVariableOp,
            std::make_pair(std::move(variableIndices), equationOp));
      } else {
        IndexSet variableIndices = accessFunction.map(IndexSet());

        writesMap.emplace(
            writtenVariableOp,
            std::make_pair(std::move(variableIndices), equationOp));
      }
    }
  }

  return mlir::success();
}

namespace
{
  class IDAPass : public mlir::modelica::impl::IDAPassBase<IDAPass>,
                  public mlir::modelica::impl::ModelSolver
  {
    public:
      using IDAPassBase::IDAPassBase;

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
      /// Add an equation to the IDA instance together with its written
      /// variable.
      mlir::LogicalResult addICModelEquation(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          IDAInstance& idaInstance,
          ScheduledEquationInstanceOp equationOp);

      /// Add an equation to the IDA instance together with its written
      /// variable.
      mlir::LogicalResult addMainModelEquation(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          const DerivativesMap& derivativesMap,
          IDAInstance& idaInstance,
          ScheduledEquationInstanceOp equationOp);

      /// Create the function that instantiates the external solvers to be used
      /// during the IC computation.
      mlir::LogicalResult createInitICSolversFunction(
          mlir::IRRewriter& rewriter,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::Location loc,
          ModelOp modelOp,
          IDAInstance* idaInstance,
          llvm::ArrayRef<VariableOp> variableOps,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          llvm::ArrayRef<SCCOp> SCCs) const;

      /// Create the function that deallocates the external solvers used during
      /// the IC computation.
      mlir::LogicalResult createDeinitICSolversFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          IDAInstance* idaInstance) const;

      /// Create the function that instantiates the external solvers to be used
      /// during the simulation.
      mlir::LogicalResult createInitMainSolversFunction(
          mlir::IRRewriter& rewriter,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::Location loc,
          ModelOp modelOp,
          IDAInstance* idaInstance,
          llvm::ArrayRef<VariableOp> variableOps,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          llvm::ArrayRef<SCCOp> SCCs) const;

      /// Create the function that deallocates the external solvers used during
      /// the simulation.
      mlir::LogicalResult createDeinitMainSolversFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          IDAInstance* idaInstance) const;

      /// Create the function that computes the initial conditions of the
      /// "initial conditions model".
      mlir::LogicalResult createSolveICModelFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::Location loc,
          IDAInstance* idaInstance,
          llvm::ArrayRef<SCCOp> SCCs,
          const llvm::DenseMap<
              ScheduledEquationInstanceOp,
              RawFunctionOp>& equationFunctions) const;

      /// Create the function that computes the initial conditions of the "main
      /// model".
      mlir::LogicalResult createCalcICFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          IDAInstance* idaInstance) const;

      /// Create the functions that calculates the values that the variables
      /// belonging to IDA will have in the next iteration.
      mlir::LogicalResult createUpdateIDAVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          IDAInstance* idaInstance) const;

      /// Create the functions that calculates the values that the variables
      /// not belonging to IDA will have in the next iteration.
      mlir::LogicalResult createUpdateNonIDAVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::Location loc,
          IDAInstance* idaInstance,
          llvm::ArrayRef<SCCOp> SCCs,
          const llvm::DenseMap<
              ScheduledEquationInstanceOp, RawFunctionOp>& equationFunctions) const;

      /// Create the function to be used to get the time reached by IDA.
      mlir::LogicalResult createGetIDATimeFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          IDAInstance* idaInstance) const;
  };
}

void IDAPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  llvm::SmallVector<ModelOp, 1> modelOps;

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

DerivativesMap& IDAPass::getDerivativesMap(ModelOp modelOp)
{
  if (auto analysis = getCachedChildAnalysis<DerivativesMap>(modelOp)) {
    return *analysis;
  }

  auto& analysis = getChildAnalysis<DerivativesMap>(modelOp);
  analysis.initialize();
  return analysis;
}

mlir::LogicalResult IDAPass::solveICModel(
    mlir::IRRewriter& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    llvm::ArrayRef<SCCOp> SCCs)
{
  LLVM_DEBUG(llvm::dbgs() << "Solving the 'initial conditions' model\n");
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

  auto idaInstance = std::make_unique<IDAInstance>(
      "ida_ic", symbolTableCollection, nullptr,
      reducedSystem, reducedDerivatives, jacobianOneSweep, debugInformation);

  // Set the start and end times.
  idaInstance->setStartTime(0);
  idaInstance->setEndTime(0);

  // Map from explicit equations to their algorithmically equivalent function.
  llvm::DenseMap<
      ScheduledEquationInstanceOp, RawFunctionOp> equationFunctions;

  if (reducedSystem) {
    LLVM_DEBUG(llvm::dbgs() << "Reduced system feature enabled\n");

    // Determine which equations can be processed by just making them explicit
    // with respect to the variable they match.
    llvm::DenseSet<ScheduledEquationInstanceOp> explicitEquations;
    llvm::SmallVector<SCCOp> cycles;
    llvm::SmallVector<ScheduledEquationInstanceOp> implicitEquations;

    for (SCCOp scc : SCCs) {
      if (scc.getCycle()) {
        // Skip the equation if it is already handled by IDA.
        cycles.push_back(scc);
      } else {
        // The content of an SCC may be modified, so we need to freeze the
        // initial list of equations.
        llvm::SmallVector<ScheduledEquationInstanceOp> equationOps;
        scc.collectEquations(equationOps);

        for (ScheduledEquationInstanceOp equationOp : equationOps) {
          if (idaInstance->hasEquation(equationOp)) {
            LLVM_DEBUG({
              llvm::dbgs() << "Equation already handled by IDA\n";
              equationOp.printInline(llvm::dbgs());
              llvm::dbgs() << "\n";
            });

            continue;
          }

          LLVM_DEBUG({
              llvm::dbgs() << "Explicitating equation\n";
              equationOp.printInline(llvm::dbgs());
              llvm::dbgs() << "\n";
          });

          auto explicitEquationOp = equationOp.cloneAndExplicitate(
              rewriter, symbolTableCollection);

          if (explicitEquationOp) {
            LLVM_DEBUG({
                llvm::dbgs() << "Add explicit equation\n";
                explicitEquationOp.printInline(llvm::dbgs());
                llvm::dbgs() << "\n";
            });

            explicitEquations.insert(explicitEquationOp);
            rewriter.eraseOp(equationOp);
          } else {
            LLVM_DEBUG(llvm::dbgs() << "Implicit equation found\n");
            implicitEquations.push_back(equationOp);
            continue;
          }
        }
      }
    }

    size_t equationFunctionsCounter = 0;

    for (ScheduledEquationInstanceOp equationOp : explicitEquations) {
      RawFunctionOp templateFunction = createEquationTemplateFunction(
          rewriter, moduleOp, symbolTableCollection,
          equationOp.getTemplate(),
          equationOp.getViewElementIndex(),
          equationOp.getIterationDirections(),
          "initial_equation_" +
              std::to_string(equationFunctionsCounter++),
          localToGlobalVariablesMap);

      equationFunctions[equationOp] = templateFunction;
    }

    // Add the implicit equations to the set of equations managed by IDA,
    // together with their written variables.

    for (ScheduledEquationInstanceOp equationOp : implicitEquations) {
      if (mlir::failed(addICModelEquation(
              symbolTableCollection, modelOp, *idaInstance, equationOp))) {
        return mlir::failure();
      }
    }

    // Add the cyclic equations to the set of equations managed by IDA,
    // together with their written variables.
    LLVM_DEBUG(llvm::dbgs() << "Add the cyclic equations\n");

    for (SCCOp scc : cycles) {
      for (ScheduledEquationInstanceOp equationOp :
           scc.getOps<ScheduledEquationInstanceOp>()) {
        if (mlir::failed(addICModelEquation(
                symbolTableCollection, modelOp, *idaInstance, equationOp))) {
          return mlir::failure();
        }
      }
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Reduced system feature disabled");

    // Add all the variables to IDA.
    for (VariableOp variableOp : variableOps) {
      LLVM_DEBUG(llvm::dbgs() << "Add algebraic variable: "
                              << variableOp.getSymName() << "\n");

      idaInstance->addAlgebraicVariable(variableOp);
    }
  }

  // If any of the remaining equations manageable by MARCO does write on a
  // variable managed by IDA, then the equation must be passed to IDA even
  // if not strictly necessary. Avoiding this would require either memory
  // duplication or a more severe restructuring of the solving
  // infrastructure, which would have to be able to split variables and
  // equations according to which runtime solver manages such variables.
  LLVM_DEBUG(llvm::dbgs() << "Add the equations writing to IDA variables\n");

  for (SCCOp scc : SCCs) {
    for (ScheduledEquationInstanceOp equationOp :
         scc.getOps<ScheduledEquationInstanceOp>()) {
      std::optional<VariableAccess> writeAccess =
          equationOp.getMatchedAccess(symbolTableCollection);

      if (!writeAccess) {
        return mlir::failure();
      }

      auto writtenVariable = writeAccess->getVariable();

      auto writtenVariableOp =
          symbolTableCollection.lookupSymbolIn<VariableOp>(
              modelOp, writtenVariable);

      if (idaInstance->hasVariable(writtenVariableOp)) {
        LLVM_DEBUG({
            llvm::dbgs() << "Add equation\n";
            equationOp.printInline(llvm::dbgs());
            llvm::dbgs() << "\n";
        });

        idaInstance->addEquation(equationOp);
      }
    }
  }

  if (mlir::failed(idaInstance->declareInstance(
          rewriter, modelOp.getLoc(), moduleOp))) {
    return mlir::failure();
  }

  if (mlir::failed(createInitICSolversFunction(
          rewriter, moduleOp, symbolTableCollection, modelOp.getLoc(), modelOp,
          idaInstance.get(), variableOps, localToGlobalVariablesMap, SCCs))) {
    return mlir::failure();
  }

  if (mlir::failed(createDeinitICSolversFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createSolveICModelFunction(
          rewriter, moduleOp, symbolTableCollection, modelOp.getLoc(),
          idaInstance.get(), SCCs, equationFunctions))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult IDAPass::addICModelEquation(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    IDAInstance& idaInstance,
    ScheduledEquationInstanceOp equationOp)
{
  LLVM_DEBUG({
      llvm::dbgs() << "Add equation\n";
      equationOp.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
  });

  idaInstance.addEquation(equationOp);

  std::optional<VariableAccess> writeAccess =
      equationOp.getMatchedAccess(symbolTableCollection);

  if (!writeAccess) {
    return mlir::failure();
  }

  auto writtenVariableOp =
      symbolTableCollection.lookupSymbolIn<VariableOp>(
          modelOp, writeAccess->getVariable());

  LLVM_DEBUG(llvm::dbgs() << "Add algebraic variable: "
                          << writtenVariableOp.getSymName() << "\n");

  idaInstance.addAlgebraicVariable(writtenVariableOp);
  return mlir::success();
}

mlir::LogicalResult IDAPass::solveMainModel(
    mlir::IRRewriter& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps,
    const DerivativesMap& derivativesMap,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    llvm::ArrayRef<SCCOp> SCCs)
{
  LLVM_DEBUG(llvm::dbgs() << "Solving the 'main' model\n");
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

  auto idaInstance = std::make_unique<IDAInstance>(
      "ida_main", symbolTableCollection, &derivativesMap,
      reducedSystem, reducedDerivatives, jacobianOneSweep, debugInformation);

  // Map from explicit equations to their algorithmically equivalent function.
  llvm::DenseMap<
      ScheduledEquationInstanceOp, RawFunctionOp> equationFunctions;

  if (reducedSystem) {
    LLVM_DEBUG(llvm::dbgs() << "Reduced system feature enabled");

    // Add the state and derivative variables.
    // All of them must always be known to IDA.

    for (VariableOp variableOp : variableOps) {
      if (auto derivative = derivativesMap.getDerivative(
              mlir::SymbolRefAttr::get(variableOp.getSymNameAttr()))) {
        assert(derivative->getNestedReferences().empty());

        auto derivativeVariableOp =
            symbolTableCollection.lookupSymbolIn<VariableOp>(
                modelOp, derivative->getRootReference());

        assert(derivativeVariableOp && "Derivative not found");

        LLVM_DEBUG(llvm::dbgs() << "Add state variable: "
                                << variableOp.getSymName() << "\n");
        idaInstance->addStateVariable(variableOp);

        LLVM_DEBUG(llvm::dbgs() << "Add derivative variable: "
                                << derivativeVariableOp.getSymName() << "\n");
        idaInstance->addDerivativeVariable(derivativeVariableOp);
      }
    }

    // Add the equations writing to variables handled by IDA.
    for (SCCOp scc : SCCs) {
      for (ScheduledEquationInstanceOp equationOp :
           scc.getOps<ScheduledEquationInstanceOp>()) {
        std::optional<VariableAccess> writeAccess =
            equationOp.getMatchedAccess(symbolTableCollection);

        if (!writeAccess) {
          LLVM_DEBUG({
              llvm::dbgs() << "Can't get write access for equation\n";
              equationOp.printInline(llvm::dbgs());
              llvm::dbgs() << "\n";
          });

          return mlir::failure();
        }

        auto writtenVariable = writeAccess->getVariable();

        auto writtenVariableOp =
            symbolTableCollection.lookupSymbolIn<VariableOp>(
                modelOp, writtenVariable);

        if (idaInstance->hasVariable(writtenVariableOp)) {
          LLVM_DEBUG({
              llvm::dbgs() << "Add equation writing to variable "
                           << writtenVariableOp.getSymName() << "\n";

              equationOp.printInline(llvm::dbgs());
              llvm::dbgs() << "\n";
          });

          idaInstance->addEquation(equationOp);
        }
      }
    }

    // Determine which equations can be processed by just making them explicit
    // with respect to the variable they match.
    llvm::DenseSet<ScheduledEquationInstanceOp> explicitEquations;
    llvm::SmallVector<SCCOp> cycles;
    llvm::SmallVector<ScheduledEquationInstanceOp> implicitEquations;

    for (SCCOp scc : SCCs) {
      if (scc.getCycle()) {
        cycles.push_back(scc);
      } else {
        // The content of an SCC may be modified, so we need to freeze the
        // initial list of equations.
        llvm::SmallVector<ScheduledEquationInstanceOp> equationOps;
        scc.collectEquations(equationOps);

        for (ScheduledEquationInstanceOp equationOp : equationOps) {
          if (idaInstance->hasEquation(equationOp)) {
            // Skip the equation if it is already handled by IDA.
            LLVM_DEBUG({
                llvm::dbgs() << "Equation already handled by IDA\n";
                equationOp.printInline(llvm::dbgs());
                llvm::dbgs() << "\n";
            });

            continue;
          }

          LLVM_DEBUG({
              llvm::dbgs() << "Explicitating equation\n";
              equationOp.printInline(llvm::dbgs());
              llvm::dbgs() << "\n";
          });

          auto explicitEquationOp =
              equationOp.cloneAndExplicitate(rewriter, symbolTableCollection);

          if (explicitEquationOp) {
            LLVM_DEBUG({
                llvm::dbgs() << "Add explicit equation\n";
                explicitEquationOp.printInline(llvm::dbgs());
            });

            explicitEquations.insert(explicitEquationOp);
            rewriter.eraseOp(equationOp);
          } else {
            LLVM_DEBUG(llvm::dbgs() << "Implicit equation found\n");
            implicitEquations.push_back(equationOp);
            continue;
          }
        }
      }
    }

    size_t equationFunctionsCounter = 0;

    for (ScheduledEquationInstanceOp equationOp : explicitEquations) {
      RawFunctionOp templateFunction = createEquationTemplateFunction(
          rewriter, moduleOp, symbolTableCollection,
          equationOp.getTemplate(),
          equationOp.getViewElementIndex(),
          equationOp.getIterationDirections(),
          "equation_" + std::to_string(equationFunctionsCounter++),
          localToGlobalVariablesMap);

      equationFunctions[equationOp] = templateFunction;
    }

    // Add the implicit equations to the set of equations managed by IDA,
    // together with their written variables.
    LLVM_DEBUG(llvm::dbgs() << "Add the implicit equations\n");

    for (ScheduledEquationInstanceOp equationOp : implicitEquations) {
      if (mlir::failed(addMainModelEquation(
              symbolTableCollection, modelOp, derivativesMap, *idaInstance,
              equationOp))) {
        return mlir::failure();
      }
    }

    // Add the cyclic equations to the set of equations managed by IDA,
    // together with their written variables.
    LLVM_DEBUG(llvm::dbgs() << "Add the cyclic equations\n");

    for (SCCOp scc : cycles) {
      for (ScheduledEquationInstanceOp equationOp :
           scc.getOps<ScheduledEquationInstanceOp>()) {
        if (mlir::failed(addMainModelEquation(
                symbolTableCollection, modelOp, derivativesMap, *idaInstance,
                equationOp))) {
          return mlir::failure();
        }
      }
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Reduced system feature disabled");

    // Add all the variables to IDA.
    for (VariableOp variableOp : variableOps) {
      auto variableName =
          mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());

      if (derivativesMap.getDerivative(variableName)) {
        LLVM_DEBUG(llvm::dbgs() << "Add state variable: "
                                << variableOp.getSymName() << "\n");
        idaInstance->addStateVariable(variableOp);
      } else if (derivativesMap.getDerivedVariable(variableName)) {
        LLVM_DEBUG(llvm::dbgs() << "Add derivative variable: "
                                << variableOp.getSymName() << "\n");
        idaInstance->addDerivativeVariable(variableOp);
      } else if (!variableOp.isReadOnly()) {
        LLVM_DEBUG(llvm::dbgs() << "Add algebraic variable: "
                                << variableOp.getSymName() << "\n");
        idaInstance->addAlgebraicVariable(variableOp);
      }
    }
  }

  // If any of the remaining equations manageable by MARCO does write on a
  // variable managed by IDA, then the equation must be passed to IDA even
  // if not strictly necessary. Avoiding this would require either memory
  // duplication or a more severe restructuring of the solving
  // infrastructure, which would have to be able to split variables and
  // equations according to which runtime solver manages such variables.
  LLVM_DEBUG(llvm::dbgs() << "Add the equations writing to IDA variables\n");

  for (SCCOp scc : SCCs) {
    for (ScheduledEquationInstanceOp equationOp :
         scc.getOps<ScheduledEquationInstanceOp>()) {
      std::optional<VariableAccess> writeAccess =
          equationOp.getMatchedAccess(symbolTableCollection);

      if (!writeAccess) {
        return mlir::failure();
      }

      auto writtenVariable = writeAccess->getVariable();

      auto writtenVariableOp =
          symbolTableCollection.lookupSymbolIn<VariableOp>(
              modelOp, writtenVariable);

      if (idaInstance->hasVariable(writtenVariableOp)) {
        LLVM_DEBUG({
            llvm::dbgs() << "Add equation\n";
            equationOp.printInline(llvm::dbgs());
            llvm::dbgs() << "\n";
        });

        idaInstance->addEquation(equationOp);
      }
    }
  }

  if (mlir::failed(idaInstance->declareInstance(
          rewriter, modelOp.getLoc(), moduleOp))) {
    return mlir::failure();
  }

  if (mlir::failed(createInitMainSolversFunction(
          rewriter, moduleOp, symbolTableCollection, modelOp.getLoc(), modelOp,
          idaInstance.get(), variableOps, localToGlobalVariablesMap, SCCs))) {
    return mlir::failure();
  }

  if (mlir::failed(createDeinitMainSolversFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createCalcICFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateIDAVariablesFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateNonIDAVariablesFunction(
          rewriter, moduleOp, symbolTableCollection, modelOp.getLoc(),
          idaInstance.get(), SCCs, equationFunctions))) {
    return mlir::failure();
  }

  if (mlir::failed(createGetIDATimeFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult IDAPass::addMainModelEquation(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    const DerivativesMap& derivativesMap,
    IDAInstance& idaInstance,
    ScheduledEquationInstanceOp equationOp)
{
  LLVM_DEBUG({
      llvm::dbgs() << "Add equation\n";
      equationOp.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
  });

  idaInstance.addEquation(equationOp);

  std::optional<VariableAccess> writeAccess =
      equationOp.getMatchedAccess(symbolTableCollection);

  if (!writeAccess) {
    return mlir::failure();
  }

  auto writtenVariable = writeAccess->getVariable();

  auto writtenVariableOp =
      symbolTableCollection.lookupSymbolIn<VariableOp>(
          modelOp, writtenVariable);

  if (derivativesMap.getDerivedVariable(writtenVariable)) {
    LLVM_DEBUG(llvm::dbgs() << "Add derivative variable: "
                            << writtenVariableOp.getSymName() << "\n");
    idaInstance.addDerivativeVariable(writtenVariableOp);
  } else if (derivativesMap.getDerivative(writtenVariable)) {
    LLVM_DEBUG(llvm::dbgs() << "Add state variable: "
                            << writtenVariableOp.getSymName() << "\n");
    idaInstance.addStateVariable(writtenVariableOp);
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Add algebraic variable: "
                            << writtenVariableOp.getSymName() << "\n");
    idaInstance.addAlgebraicVariable(writtenVariableOp);
  }

  return mlir::success();
}

mlir::LogicalResult IDAPass::createInitICSolversFunction(
    mlir::IRRewriter& rewriter,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::Location loc,
    ModelOp modelOp,
    IDAInstance* idaInstance,
    llvm::ArrayRef<VariableOp> variableOps,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    llvm::ArrayRef<SCCOp> SCCs) const
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::simulation::FunctionOp>(
      loc, "initICSolvers",
      rewriter.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  if (mlir::failed(idaInstance->initialize(rewriter, loc))) {
    return mlir::failure();
  }

  if (mlir::failed(idaInstance->configure(
          rewriter, loc, moduleOp, modelOp, variableOps,
          localToGlobalVariablesMap, SCCs))) {
    return mlir::failure();
  }

  rewriter.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createDeinitICSolversFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "deinitICSolvers",
      builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Deallocate the IDA solver.
  if (mlir::failed(idaInstance->deleteInstance(builder, loc))) {
    return mlir::failure();
  }

  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createInitMainSolversFunction(
    mlir::IRRewriter& rewriter,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::Location loc,
    ModelOp modelOp,
    IDAInstance* idaInstance,
    llvm::ArrayRef<VariableOp> variableOps,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    llvm::ArrayRef<SCCOp> SCCs) const
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::simulation::FunctionOp>(
      loc, "initMainSolvers",
      rewriter.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  if (mlir::failed(idaInstance->initialize(rewriter, loc))) {
    return mlir::failure();
  }

  if (mlir::failed(idaInstance->configure(
          rewriter, loc, moduleOp, modelOp, variableOps,
          localToGlobalVariablesMap, SCCs))) {
    return mlir::failure();
  }

  rewriter.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createDeinitMainSolversFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "deinitMainSolvers",
      builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  if (mlir::failed(idaInstance->deleteInstance(builder, loc))) {
    return mlir::failure();
  }

  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createSolveICModelFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::Location loc,
    IDAInstance* idaInstance,
    llvm::ArrayRef<SCCOp> SCCs,
    const llvm::DenseMap<
        ScheduledEquationInstanceOp, RawFunctionOp>& equationFunctions) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "solveICModel",
      builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Compute the initial conditions for the equations managed by IDA.
  if (mlir::failed(idaInstance->performCalcIC(builder, loc))) {
    return mlir::failure();
  }

  // Call the equation functions for the equations managed internally.
  for (SCCOp scc : SCCs) {
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

  builder.setInsertionPointToEnd(entryBlock);
  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createCalcICFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "calcIC",
      builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  if (mlir::failed(idaInstance->performCalcIC(builder, loc))) {
    return mlir::failure();
  }

  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createUpdateIDAVariablesFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "updateIDAVariables",
      builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  if (mlir::failed(idaInstance->performStep(builder, loc))) {
    return mlir::failure();
  }

  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createUpdateNonIDAVariablesFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::Location loc,
    IDAInstance* idaInstance,
    llvm::ArrayRef<SCCOp> SCCs,
    const llvm::DenseMap<
        ScheduledEquationInstanceOp, RawFunctionOp>& equationFunctions) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "updateNonIDAVariables",
      builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Call the equation functions for the equations managed internally.
  for (SCCOp scc : SCCs) {
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

  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createGetIDATimeFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "getIDATime",
      builder.getFunctionType(std::nullopt, builder.getF64Type()));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  mlir::Value time = idaInstance->getCurrentTime(builder, loc);
  builder.create<mlir::simulation::ReturnOp>(loc, time);
  return mlir::success();
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
