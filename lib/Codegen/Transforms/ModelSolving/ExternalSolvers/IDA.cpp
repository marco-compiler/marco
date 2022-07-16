#include "marco/Codegen/Transforms/ModelSolving/ExternalSolvers/IDASolver.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "marco/Codegen/Utils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <queue>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  IDAVariable::IDAVariable(unsigned int argNumber, IDAVariableType type)
    : argNumber(argNumber), type(type)
  {
  }

  IDASolver::IDASolver(
      mlir::TypeConverter* typeConverter,
      const DerivativesMap& derivativesMap,
      IDAOptions options,
      double startTime,
      double endTime,
      double timeStep)
    : ExternalSolver(typeConverter),
      derivativesMap(&derivativesMap),
      enabled(true),
      options(std::move(options)),
      startTime(startTime),
      endTime(endTime),
      timeStep(timeStep)
  {
    assert(startTime <= endTime);
    assert(timeStep > 0);
  }

  bool IDASolver::isEnabled() const
  {
    return enabled;
  }

  void IDASolver::setEnabled(bool status)
  {
    enabled = status;
  }

  bool IDASolver::containsEquation(ScheduledEquation* equation) const
  {
    return equations.find(equation) != equations.end();
  }

  mlir::Type IDASolver::getRuntimeDataType(mlir::MLIRContext* context)
  {
    std::vector<mlir::Type> structTypes;

    // The IDA instance is the first element within the struct
    structTypes.push_back(getTypeConverter()->convertType(mlir::ida::InstanceType::get(context)));

    // Then add an IDA variable for each algebraic and state variable
    for (const auto& variable : managedVariables) {
      if (variable.type == IDAVariableType::ALGEBRAIC || variable.type == IDAVariableType::STATE) {
        structTypes.push_back(getTypeConverter()->convertType(mlir::ida::VariableType::get(context)));
      }
    }

    return mlir::LLVM::LLVMStructType::getLiteral(context, structTypes);
  }

  bool IDASolver::hasVariable(mlir::Value variable) const
  {
    auto argNumber = variable.cast<mlir::BlockArgument>().getArgNumber();

    auto it = llvm::find_if(managedVariables, [&](const auto& managedVariable) {
      return managedVariable.argNumber == argNumber;
    });

    return it != managedVariables.end();
  }

  void IDASolver::addVariable(mlir::Value variable)
  {
    assert(variable.isa<mlir::BlockArgument>());

    if (!hasVariable(variable)) {
      auto argNumber = variable.cast<mlir::BlockArgument>().getArgNumber();

      if (derivativesMap->isDerivative(argNumber)) {
        managedVariables.emplace_back(argNumber, IDAVariableType::DERIVATIVE);
      } else if (derivativesMap->hasDerivative(argNumber)) {
        managedVariables.emplace_back(argNumber, IDAVariableType::STATE);
      } else {
        managedVariables.emplace_back(argNumber, IDAVariableType::ALGEBRAIC);
      }
    }
  }

  bool IDASolver::hasEquation(ScheduledEquation* equation) const
  {
    return llvm::find(equations, equation) != equations.end();
  }

  void IDASolver::addEquation(ScheduledEquation* equation)
  {
    equations.emplace(equation);
  }

  mlir::Value IDASolver::materializeTargetConversion(
      mlir::OpBuilder& builder, mlir::Value value)
  {
    auto convertedType = getTypeConverter()->convertType(value.getType());
    return getTypeConverter()->materializeTargetConversion(builder, value.getLoc(), convertedType, value);
  }

  mlir::Value IDASolver::loadRuntimeData(
      mlir::OpBuilder& builder, mlir::Value runtimeDataPtr)
  {
    assert(runtimeDataPtr.getType().isa<mlir::LLVM::LLVMPointerType>());
    return builder.create<mlir::LLVM::LoadOp>(runtimeDataPtr.getLoc(), runtimeDataPtr);
  }

  void IDASolver::storeRuntimeData(
      mlir::OpBuilder& builder, mlir::Value runtimeDataPtr, mlir::Value value)
  {
    assert(runtimeDataPtr.getType().isa<mlir::LLVM::LLVMPointerType>());
    assert(runtimeDataPtr.getType().cast<mlir::LLVM::LLVMPointerType>().getElementType() == value.getType());

    builder.create<mlir::LLVM::StoreOp>(value.getLoc(), value, runtimeDataPtr);
  }

  mlir::Value IDASolver::getValueFromRuntimeData(
      mlir::OpBuilder& builder, mlir::Value structValue, mlir::Type type, unsigned int position)
  {
    auto loc = structValue.getLoc();

    assert(structValue.getType().isa<mlir::LLVM::LLVMStructType>() && "Not an LLVM struct");
    auto structType = structValue.getType().cast<mlir::LLVM::LLVMStructType>();
    auto structTypes = structType.getBody();
    assert (position < structTypes.size() && "LLVM struct: index is out of bounds");

    mlir::Value var = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, structTypes[position], structValue, builder.getIndexArrayAttr(position));

    return getTypeConverter()->materializeSourceConversion(builder, loc, type, var);
  }

  mlir::Value IDASolver::getIDAInstance(
      mlir::OpBuilder& builder, mlir::Value runtimeData)
  {
    return getValueFromRuntimeData(
        builder, runtimeData,
        mlir::ida::InstanceType::get(builder.getContext()),
        idaInstancePosition);
  }

  mlir::Value IDASolver::getIDAVariable(
      mlir::OpBuilder& builder, mlir::Value runtimeData, unsigned int position)
  {
    return getValueFromRuntimeData(
        builder, runtimeData,
        mlir::ida::VariableType::get(builder.getContext()),
        position + variablesOffset);
  }

  mlir::Value IDASolver::setIDAInstance(
      mlir::OpBuilder& builder, mlir::Value runtimeData, mlir::Value instance)
  {
    assert(instance.getType().isa<mlir::ida::InstanceType>());

    return builder.create<mlir::LLVM::InsertValueOp>(
        instance.getLoc(),
        runtimeData,
        materializeTargetConversion(builder, instance),
        builder.getIndexArrayAttr(idaInstancePosition));
  }

  mlir::Value IDASolver::setIDAVariable(
      mlir::OpBuilder& builder, mlir::Value runtimeData, unsigned int position, mlir::Value variable)
  {
    assert(variable.getType().isa<mlir::ida::VariableType>());

    return builder.create<mlir::LLVM::InsertValueOp>(
        variable.getLoc(),
        runtimeData,
        materializeTargetConversion(builder, variable),
        builder.getIndexArrayAttr(position + variablesOffset));
  }

  mlir::LogicalResult IDASolver::processInitFunction(
      mlir::OpBuilder& builder,
      mlir::Value runtimeDataPtr,
      mlir::func::FuncOp initFunction,
      mlir::ValueRange variables,
      const Model<ScheduledEquationsBlock>& model)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto module = initFunction->getParentOfType<mlir::ModuleOp>();

    auto terminator = mlir::cast<mlir::func::ReturnOp>(initFunction.getBody().back().getTerminator());
    builder.setInsertionPoint(terminator);

    // Create the IDA instance.
    // To do so, we need to first compute the total number of scalar variables that IDA
    // has to manage. Such number is equal to the number of scalar equations.
    size_t numberOfScalarEquations = 0;

    for (const auto& equation : equations) {
      numberOfScalarEquations += equation->getIterationRanges().flatSize();
    }

    mlir::Value idaInstance = builder.create<mlir::ida::CreateOp>(
        initFunction.getLoc(), builder.getI64IntegerAttr(numberOfScalarEquations));

    builder.create<mlir::ida::SetStartTimeOp>(idaInstance.getLoc(), idaInstance, builder.getF64FloatAttr(startTime));
    builder.create<mlir::ida::SetEndTimeOp>(idaInstance.getLoc(), idaInstance, builder.getF64FloatAttr(endTime));

    if (options.equidistantTimeGrid) {
      builder.create<mlir::ida::SetTimeStepOp>(idaInstance.getLoc(), idaInstance, builder.getF64FloatAttr(timeStep));
    }

    // Store the IDA instance into the runtime data structure
    mlir::Value runtimeData = loadRuntimeData(builder, runtimeDataPtr);
    runtimeData = setIDAInstance(builder, runtimeData, idaInstance);
    storeRuntimeData(builder, runtimeDataPtr, runtimeData);

    // Add the variables to IDA
    if (auto res = addVariablesToIDA(builder, module, runtimeDataPtr, variables); mlir::failed(res)) {
      return res;
    }

    // Add the equations to IDA
    if (auto res = addEquationsToIDA(builder, runtimeDataPtr, model); mlir::failed(res)) {
      return res;
    }

    // Initialize the IDA instance
    builder.create<mlir::ida::InitOp>(initFunction.getLoc(), idaInstance);

    return mlir::success();
  }

  mlir::LogicalResult IDASolver::addVariablesToIDA(
      mlir::OpBuilder& builder,
      mlir::ModuleOp module,
      mlir::Value runtimeDataPtr,
      mlir::ValueRange variables)
  {
    mlir::Value runtimeData = loadRuntimeData(builder, runtimeDataPtr);
    mlir::Value idaInstance = getIDAInstance(builder, runtimeData);

    unsigned int getterFunctionCounter = 0;
    unsigned int setterFunctionCounter = 0;
    unsigned int idaVariableIndex = 0;

    // Map between the original variable argument numbers and the IDA state variables
    std::map<unsigned int, mlir::Value> idaStateVariables;

    for (const auto& managedVariable : managedVariables) {
      if (managedVariable.type == IDAVariableType::DERIVATIVE) {
        continue;
      }

      mlir::Value variable = variables[managedVariable.argNumber];
      auto arrayType = variable.getType().cast<ArrayType>();
      assert(arrayType.hasConstantShape());

      std::vector<int64_t> dimensions;

      if (arrayType.isScalar()) {
        // In case of scalar variables, the shape of the array would be empty
        // but IDA needs to see a single dimension of value 1.
        dimensions.push_back(1);
      } else {
        auto shape = arrayType.getShape();
        dimensions.insert(dimensions.end(), shape.begin(), shape.end());
      }

      std::string getterName = getUniqueSymbolName(module, [&]() {
        return "ida_getter_" + std::to_string(getterFunctionCounter++);
      });

      if (auto res = createGetterFunction(builder, module, variable.getLoc(), arrayType, getterName); mlir::failed(res)) {
        return res;
      }

      std::string setterName = getUniqueSymbolName(module, [&]() {
        return "ida_setter_" + std::to_string(setterFunctionCounter++);
      });

      if (auto res = createSetterFunction(builder, module, variable.getLoc(), arrayType, setterName); mlir::failed(res)) {
        return res;
      }

      if (managedVariable.type == IDAVariableType::ALGEBRAIC) {
        mlir::Value idaVariable = builder.create<mlir::ida::AddAlgebraicVariableOp>(
            variable.getLoc(),
            idaInstance,
            variable,
            builder.getI64ArrayAttr(dimensions),
            getterName,
            setterName);

        mappedVariables[managedVariable.argNumber] = idaVariableIndex;
        runtimeData = setIDAVariable(builder, runtimeData, idaVariableIndex, idaVariable);

        ++idaVariableIndex;
      }

      if (managedVariable.type == IDAVariableType::STATE) {
        mlir::Value idaStateVariable = builder.create<mlir::ida::AddStateVariableOp>(
            variable.getLoc(),
            idaInstance,
            variable,
            builder.getI64ArrayAttr(dimensions),
            getterName,
            setterName);

        mappedVariables[managedVariable.argNumber] = idaVariableIndex;
        idaStateVariables[managedVariable.argNumber] = idaStateVariable;

        runtimeData = setIDAVariable(builder, runtimeData, idaVariableIndex, idaStateVariable);

        const auto& derivative = derivativesMap->getDerivative(managedVariable.argNumber);
        mappedVariables[derivative] = idaVariableIndex;

        ++idaVariableIndex;
      }
    }

    for (const auto& managedVariable : llvm::enumerate(managedVariables)) {
      if (managedVariable.value().type == IDAVariableType::DERIVATIVE) {
        mlir::Value derVar = variables[managedVariable.value().argNumber];
        auto arrayType = derVar.getType().cast<ArrayType>();
        assert(arrayType.hasConstantShape());

        std::vector<int64_t> dimensions;

        if (arrayType.isScalar()) {
          // In case of scalar variables, the shape of the array would be empty
          // but IDA needs to see a single dimension of value 1.
          dimensions.push_back(1);
        } else {
          auto shape = arrayType.getShape();
          dimensions.insert(dimensions.end(), shape.begin(), shape.end());
        }

        auto state = derivativesMap->getDerivedVariable(managedVariable.value().argNumber);

        auto idaStateVariableIt = idaStateVariables.find(state);
        assert(idaStateVariableIt != idaStateVariables.end());
        auto idaStateVariable = idaStateVariableIt->second;

        std::string getterName = getUniqueSymbolName(module, [&]() {
          return "ida_getter_" + std::to_string(getterFunctionCounter++);
        });

        if (auto res = createGetterFunction(builder, module, derVar.getLoc(), derVar.getType(), getterName); mlir::failed(res)) {
          return res;
        }

        std::string setterName = getUniqueSymbolName(module, [&]() {
          return "ida_setter_" + std::to_string(setterFunctionCounter++);
        });

        if (auto res = createSetterFunction(builder, module, derVar.getLoc(), derVar.getType(), setterName); mlir::failed(res)) {
          return res;
        }

        builder.create<mlir::ida::SetDerivativeOp>(
            derVar.getLoc(),
            idaInstance,
            idaStateVariable,
            derVar,
            getterName,
            setterName);
      }
    }

    storeRuntimeData(builder, runtimeDataPtr, runtimeData);

    return mlir::success();
  }

  mlir::LogicalResult IDASolver::createGetterFunction(
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
        std::max((unsigned int) 1, variableArrayType.getRank()));

    mlir::Block* entryBlock = getterOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto receivedIndices = getterOp.getVariableIndices().take_front(variableArrayType.getRank());
    mlir::Value result = builder.create<LoadOp>(loc, getterOp.getVariable(), receivedIndices);

    if (auto requestedResultType = getterOp.getFunctionType().getResult(0); result.getType() != requestedResultType) {
      result = builder.create<CastOp>(loc, requestedResultType, result);
    }

    builder.create<mlir::ida::ReturnOp>(loc, result);
    return mlir::success();
  }

  mlir::LogicalResult IDASolver::createSetterFunction(
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
        std::max((unsigned int) 1, variableArrayType.getRank()));

    mlir::Block* entryBlock = setterOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto receivedIndices = setterOp.getVariableIndices().take_front(variableArrayType.getRank());
    mlir::Value value = setterOp.getValue();

    if (auto requestedValueType = variableArrayType.getElementType(); value.getType() != requestedValueType) {
      value = builder.create<CastOp>(loc, requestedValueType, setterOp.getValue());
    }

    builder.create<StoreOp>(loc, value, setterOp.getVariable(), receivedIndices);
    builder.create<mlir::ida::ReturnOp>(loc);

    return mlir::success();
  }

  mlir::LogicalResult IDASolver::addEquationsToIDA(
      mlir::OpBuilder& builder,
      mlir::Value runtimeDataPtr,
      const Model<ScheduledEquationsBlock>& model)
  {
    auto module = model.getOperation()->getParentOfType<mlir::ModuleOp>();

    mlir::Value runtimeData = loadRuntimeData(builder, runtimeDataPtr);
    mlir::Value idaInstance = getIDAInstance(builder, runtimeData);

    // Substitute the accesses to non-IDA variables with the equations writing in such variables
    std::vector<std::unique_ptr<ScheduledEquation>> independentEquations;

    // First create the writes map, that is the knowledge of which equation writes into a variable and in which indices.
    // The variables are mapped by their argument number.
    std::multimap<unsigned int, std::pair<IndexSet, ScheduledEquation*>> writesMap;

    for (const auto& equationsBlock : model.getScheduledBlocks()) {
      for (const auto& equation : *equationsBlock) {
        if (equations.find(equation.get()) != equations.end()) {
          // Ignore the equation if it is already managed by IDA
          continue;
        }

        const auto& write = equation->getWrite();
        auto varPosition = write.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
        auto writtenIndices = write.getAccessFunction().map(equation->getIterationRanges());
        writesMap.emplace(varPosition, std::make_pair(writtenIndices, equation.get()));
      }
    }

    // The equations we are operating on
    std::queue<std::unique_ptr<ScheduledEquation>> processedEquations;

    for (const auto& equation : equations) {
      auto clone = Equation::build(equation->cloneIR(), equation->getVariables());

      auto matchedClone = std::make_unique<MatchedEquation>(
          std::move(clone), equation->getIterationRanges(), equation->getWrite().getPath());

      auto scheduledClone = std::make_unique<ScheduledEquation>(
          std::move(matchedClone), equation->getIterationRanges(), equation->getSchedulingDirection());

      processedEquations.push(std::move(scheduledClone));
    }

    while (!processedEquations.empty()) {
      auto& equation = processedEquations.front();
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

        auto readIndices = access.getAccessFunction().map(equation->getIterationRanges());
        auto varPosition = access.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
        auto writingEquations = llvm::make_range(writesMap.equal_range(varPosition));

        for (const auto& entry : writingEquations) {
          ScheduledEquation* writingEquation = entry.second.second;
          auto writtenVariableIndices = IndexSet(entry.second.first);

          if (!writtenVariableIndices.overlaps(readIndices)) {
            continue;
          }

          atLeastOneAccessReplaced = true;

          auto clone = Equation::build(equation->cloneIR(), equation->getVariables());

          auto explicitWritingEquation = writingEquation->cloneIRAndExplicitate(builder);
          TemporaryEquationGuard guard(*explicitWritingEquation);

          auto iterationRanges = explicitWritingEquation->getIterationRanges(); //todo: check ragged case
          for(auto range : llvm::make_range(iterationRanges.rangesBegin(), iterationRanges.rangesEnd()))
          {
            auto res = explicitWritingEquation->replaceInto(
                builder, IndexSet(range), *clone, access.getAccessFunction(), access.getPath());

            if (mlir::failed(res)) {
              return res;
            }
          }

          // Add the equation with the replaced access
          auto readAccessIndices = access.getAccessFunction().inverseMap(
              IndexSet(writtenVariableIndices),
              IndexSet(equation->getIterationRanges()));

          auto newEquationIndices = readAccessIndices.intersect(equation->getIterationRanges());

          for (const auto& range : llvm::make_range(newEquationIndices.rangesBegin(), newEquationIndices.rangesEnd())) {
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
    assert(llvm::none_of(independentEquations, [&](const auto& equation) {
      return llvm::any_of(equation->getAccesses(), [&](const auto& access) {
        auto argNumber = access.getVariable()->getValue().template cast<mlir::BlockArgument>().getArgNumber();

        auto it = llvm::find_if(managedVariables, [&](const auto& managedVariable) {
          return managedVariable.argNumber == argNumber;
        });

        return it == managedVariables.end();
      });
    }) && "Some non-IDA variables have not been replaced");

    // The accesses to non-IDA variables have been replaced. Now we can proceed to create the
    // residual and jacobian functions.

    auto equationVariables = model.getVariables().getValues();

    size_t residualFunctionsCounter = 0;
    size_t jacobianFunctionsCounter = 0;
    size_t partialDerTemplatesCounter = 0;

    for (const auto& equation : independentEquations) 
    {
      auto iterationRanges = equation->getIterationRanges(); //todo: check ragged case
      for(auto ranges : llvm::make_range(iterationRanges.rangesBegin(), iterationRanges.rangesEnd()))
      {
        std::vector<mlir::Attribute> rangesAttr;

        for (size_t i = 0; i < ranges.rank(); ++i) {
          rangesAttr.push_back(builder.getI64ArrayAttr({ ranges[i].getBegin(), ranges[i].getEnd() }));
        }

        auto idaEquation = builder.create<mlir::ida::AddEquationOp>(
            equation->getOperation().getLoc(),
            idaInstance,
            builder.getArrayAttr(rangesAttr));

        if (auto res = addVariableAccessesInfoToIDA(builder, runtimeDataPtr, *equation, idaEquation); mlir::failed(res)) {
          return res;
        }

        // Create the residual function
        std::string residualFunctionName = getUniqueSymbolName(module, [&]() {
          return "ida_residualFunction_" + std::to_string(residualFunctionsCounter++);
        });

        if (auto res = createResidualFunction(builder, *equation, equationVariables, idaEquation, residualFunctionName); mlir::failed(res)) {
          return res;
        }

        builder.create<mlir::ida::AddResidualOp>(equation->getOperation().getLoc(), idaInstance, idaEquation, residualFunctionName);

        // Create the partial derivative template
        std::string partialDerTemplateName = getUniqueSymbolName(module, [&]() {
          return "ida_pder_" + std::to_string(partialDerTemplatesCounter++);
        });

        if (auto res = createPartialDerTemplateFunction(builder, *equation, equationVariables, partialDerTemplateName); mlir::failed(res)) {
          return res;
        }

        // Create the Jacobian functions
        for (const auto& variable : managedVariables) {
          if (variable.type == IDAVariableType::DERIVATIVE) {
            // If the variable is a derivative, then skip the creation of the Jacobian functions
            // because it is already handled when encountering the state variable through the
            // 'alpha' parameter set into the derivative seed.
            continue;
          }

          mlir::Value var = equationVariables[variable.argNumber];

          std::string jacobianFunctionName = getUniqueSymbolName(module, [&]() {
            return "ida_jacobianFunction_" + std::to_string(jacobianFunctionsCounter++);
          });

          if (auto res = createJacobianFunction(builder, *equation, equationVariables, jacobianFunctionName, var, partialDerTemplateName); mlir::failed(res)) {
            return res;
          }

          builder.create<mlir::ida::AddJacobianOp>(
              equation->getOperation().getLoc(),
              idaInstance,
              idaEquation,
              getIDAVariable(builder, runtimeData, mappedVariables[variable.argNumber]),
              jacobianFunctionName);
        }
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult IDASolver::addVariableAccessesInfoToIDA(
      mlir::OpBuilder& builder,
      mlir::Value runtimeDataPtr,
      const Equation& equation,
      mlir::Value idaEquation)
  {
    mlir::Value runtimeData = loadRuntimeData(builder, runtimeDataPtr);
    mlir::Value idaInstance = getIDAInstance(builder, runtimeData);

    for (const auto& access : equation.getAccesses()) {
      auto argNumber = access.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
      auto idaVariableIndex = mappedVariables.find(argNumber);
      assert(idaVariableIndex != mappedVariables.end());

      mlir::Value idaVariable = getIDAVariable(builder, runtimeData, idaVariableIndex->second);

      const auto& accessFunction = access.getAccessFunction();
      std::vector<mlir::AffineExpr> expressions;

      for (const auto& dimensionAccess : accessFunction) {
        if (dimensionAccess.isConstantAccess()) {
          expressions.push_back(mlir::getAffineConstantExpr(dimensionAccess.getPosition(), builder.getContext()));
        } else {
          auto baseAccess = mlir::getAffineDimExpr(dimensionAccess.getInductionVariableIndex(), builder.getContext());
          auto withOffset = baseAccess + dimensionAccess.getOffset();
          expressions.push_back(withOffset);
        }
      }

      builder.create<mlir::ida::AddVariableAccessOp>(
          equation.getOperation().getLoc(),
          idaInstance, idaEquation, idaVariable,
          mlir::AffineMap::get(accessFunction.size(), 0, expressions, builder.getContext()));
    }

    return mlir::success();
  }

  mlir::LogicalResult IDASolver::createResidualFunction(
      mlir::OpBuilder& builder,
      const Equation& equation,
      mlir::ValueRange variables,
      mlir::Value idaEquation,
      llvm::StringRef residualFunctionName)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(equation.getOperation()->getParentOfType<mlir::ModuleOp>().getBody());

    // Pass only the variables that are managed by IDA
    auto filteredOriginalVariables = filterByManagedVariables(variables);

    // Create the residual function
    auto residualFunction = builder.create<mlir::ida::ResidualFunctionOp>(
        equation.getOperation().getLoc(),
        residualFunctionName,
        RealType::get(builder.getContext()),
        mlir::ValueRange(filteredOriginalVariables).getTypes(),
        equation.getNumOfIterationVars(),
        RealType::get(builder.getContext()));

    assert(residualFunction.getBodyRegion().empty());
    mlir::Block* bodyBlock = residualFunction.addEntryBlock();
    builder.setInsertionPointToStart(bodyBlock);

    mlir::BlockAndValueMapping mapping;

    // Map the model variables
    auto mappedVars = residualFunction.getVariables();
    assert(filteredOriginalVariables.size() == mappedVars.size());

    for (const auto& [original, mapped] : llvm::zip(filteredOriginalVariables, mappedVars)) {
      mapping.map(original, mapped);
    }

    // Map the iteration variables
    auto originalInductions = equation.getInductionVariables();
    auto mappedInductions = residualFunction.getEquationIndices();

    // Scalar equations have zero concrete values, but yet they show a fake induction variable.
    // The same happens with equations having implicit iteration variables (which originate
    // from array assignments).
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

    auto clonedTerminator = mlir::cast<EquationSidesOp>(residualFunction.getBodyRegion().back().getTerminator());

    assert(clonedTerminator.getLhsValues().size() == 1);
    assert(clonedTerminator.getRhsValues().size() == 1);

    mlir::Value lhs = clonedTerminator.getLhsValues()[0];
    mlir::Value rhs = clonedTerminator.getRhsValues()[0];

    if (lhs.getType().isa<ArrayType>()) {
      std::vector<mlir::Value> indices(
          std::next(mappedInductions.begin(), originalInductions.size()),
          mappedInductions.end());

      lhs = builder.create<LoadOp>(lhs.getLoc(), lhs, indices);
      assert((lhs.getType().isa<mlir::IndexType, BooleanType, IntegerType, RealType>()));
    }

    if (rhs.getType().isa<ArrayType>()) {
      std::vector<mlir::Value> indices(
          std::next(mappedInductions.begin(), originalInductions.size()),
          mappedInductions.end());

      rhs = builder.create<LoadOp>(rhs.getLoc(), rhs, indices);
      assert((rhs.getType().isa<mlir::IndexType, BooleanType, IntegerType, RealType>()));
    }

    mlir::Value difference = builder.create<SubOp>(residualFunction.getLoc(), RealType::get(builder.getContext()), rhs, lhs);
    builder.create<mlir::ida::ReturnOp>(difference.getLoc(), difference);

    auto lhsOp = clonedTerminator.getLhs().getDefiningOp<EquationSideOp>();
    auto rhsOp = clonedTerminator.getRhs().getDefiningOp<EquationSideOp>();
    clonedTerminator.erase();
    lhsOp.erase();
    rhsOp.erase();

    return mlir::success();
  }

  mlir::LogicalResult IDASolver::createPartialDerTemplateFunction(
      mlir::OpBuilder& builder,
      const Equation& equation,
      mlir::ValueRange equationVariables,
      llvm::StringRef templateName)
  {
    auto partialDerTemplate = createPartialDerTemplateFromEquation(
        builder, equation, equationVariables, templateName);

    // Add the time to the input members (and signature)
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(partialDerTemplate.bodyBlock());

    auto timeMember = builder.create<MemberCreateOp>(
        partialDerTemplate.getLoc(),
        "time",
        MemberType::get(builder.getContext(), RealType::get(builder.getContext()), llvm::None, false, IOProperty::input),
        llvm::None);

    mlir::Value time = builder.create<MemberLoadOp>(timeMember.getLoc(), timeMember);

    std::vector<mlir::Type> args;
    args.push_back(timeMember.getMemberType().unwrap());

    for (auto type : partialDerTemplate.getFunctionType().getInputs()) {
      args.push_back(type);
    }

    partialDerTemplate->setAttr(
        partialDerTemplate.getFunctionTypeAttrName(),
        mlir::TypeAttr::get(builder.getFunctionType(args, partialDerTemplate.getFunctionType().getResults())));

    // Replace the TimeOp with the newly created member
    partialDerTemplate.walk([&](TimeOp timeOp) {
      timeOp.replaceAllUsesWith(time);
      timeOp.erase();
    });

    return mlir::success();
  }

  FunctionOp IDASolver::createPartialDerTemplateFromEquation(
      mlir::OpBuilder& builder,
      const Equation& equation,
      mlir::ValueRange originalVariables,
      llvm::StringRef templateName)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(equation.getOperation()->getParentOfType<mlir::ModuleOp>().getBody());
    auto loc = equation.getOperation().getLoc();

    std::string functionOpName = templateName.str() + "_base";

    // Pass only the variables that are managed by IDA
    auto filteredOriginalVariables = filterByManagedVariables(originalVariables);

    // The arguments of the base function contain both the variables and the inductions
    llvm::SmallVector<mlir::Type, 6> argsTypes;

    for (auto type : mlir::ValueRange(filteredOriginalVariables).getTypes()) {
      if (auto arrayType = type.dyn_cast<ArrayType>(); arrayType && arrayType.isScalar()) {
        argsTypes.push_back(arrayType.getElementType());
      } else {
        argsTypes.push_back(type);
      }
    }

    for (size_t i = 0; i < equation.getNumOfIterationVars(); ++i) {
      argsTypes.push_back(builder.getIndexType());
    }

    // Create the function to be derived
    auto functionOp = builder.create<FunctionOp>(
        loc,
        functionOpName,
        builder.getFunctionType(argsTypes, RealType::get(builder.getContext())));

    // Start the body of the function
    mlir::Block* entryBlock = builder.createBlock(&functionOp.getBody());
    builder.setInsertionPointToStart(entryBlock);

    // Create the input members and map them to the original variables (and inductions)
    mlir::BlockAndValueMapping mapping;

    for (auto originalVar : llvm::enumerate(filteredOriginalVariables)) {
      auto memberType = MemberType::wrap(originalVar.value().getType(), false, IOProperty::input);
      auto memberOp = builder.create<MemberCreateOp>(loc, "var" + std::to_string(originalVar.index()), memberType, llvm::None);
      mapping.map(originalVar.value(), memberOp);
    }

    llvm::SmallVector<mlir::Value, 3> inductions;

    for (size_t i = 0; i < equation.getNumOfIterationVars(); ++i) {
      auto memberType = MemberType::wrap(builder.getIndexType(), false, IOProperty::input);
      auto memberOp = builder.create<MemberCreateOp>(loc, "ind" + std::to_string(i), memberType, llvm::None);
      inductions.push_back(memberOp);
    }

    // Create the output member, that is the difference between its equation right-hand side value and its
    // left-hand side value.
    auto originalTerminator = mlir::cast<EquationSidesOp>(equation.getOperation().bodyBlock()->getTerminator());
    assert(originalTerminator.getLhsValues().size() == 1);
    assert(originalTerminator.getRhsValues().size() == 1);

    auto outputMember = builder.create<MemberCreateOp>(
        loc, "out",
        MemberType::wrap(RealType::get(builder.getContext()), false, IOProperty::output),
        llvm::None);

    // Now that all the members have been created, we can load the input members and the inductions
    for (auto originalVar : filteredOriginalVariables) {
      auto mappedVar = builder.create<MemberLoadOp>(loc, mapping.lookup(originalVar));
      mapping.map(originalVar, mappedVar);
    }

    for (auto& induction : inductions) {
      induction = builder.create<MemberLoadOp>(loc, induction);
    }

    auto explicitEquationInductions = equation.getInductionVariables();

    for (const auto& originalInduction : llvm::enumerate(explicitEquationInductions)) {
      assert(originalInduction.index() < inductions.size());
      mapping.map(originalInduction.value(), inductions[originalInduction.index()]);
    }

    // Clone the original operations
    for (auto& op : equation.getOperation().bodyBlock()->getOperations()) {
      if (auto loadOp = mlir::dyn_cast<LoadOp>(op)) {
        // We need to check if the load operation is performed on a model variable.
        // Those variables are in fact created inside the function by means of MemberCreateOps,
        // and if such variable is a scalar one there would be a load operation wrongly operating
        // on a scalar value.

        if (loadOp.getArray().getType().cast<ArrayType>().isScalar()) {
          if (auto memberLoadOp = mapping.lookup(loadOp.getArray()).getDefiningOp<MemberLoadOp>()) {
            mapping.map(loadOp.getResult(), memberLoadOp.getResult());
            continue;
          }
        }
      }

      builder.clone(op, mapping);
    }

    auto terminator = mlir::cast<EquationSidesOp>(functionOp.bodyBlock()->getTerminator());
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

    auto result = builder.create<SubOp>(loc, RealType::get(builder.getContext()), rhs, lhs);
    builder.create<MemberStoreOp>(loc, outputMember, result);

    auto lhsOp = terminator.getLhs().getDefiningOp<EquationSideOp>();
    auto rhsOp = terminator.getRhs().getDefiningOp<EquationSideOp>();
    terminator.erase();
    lhsOp.erase();
    rhsOp.erase();

    // Create the derivative template
    ForwardAD forwardAD;
    auto derTemplate = forwardAD.createPartialDerTemplateFunction(builder, loc, functionOp, templateName);
    functionOp.erase();
    return derTemplate;
  }

  mlir::LogicalResult IDASolver::createJacobianFunction(
      mlir::OpBuilder& builder,
      const Equation& equation,
      mlir::ValueRange equationVariables,
      llvm::StringRef jacobianFunctionName,
      mlir::Value independentVariable,
      llvm::StringRef partialDerTemplateName)
  {
    auto loc = equation.getOperation().getLoc();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(equation.getOperation()->getParentOfType<mlir::ModuleOp>().getBody());

    // Pass only the variables that are managed by IDA
    auto filteredOriginalVariables = filterByManagedVariables(equationVariables);

    auto jacobianFunction = builder.create<mlir::ida::JacobianFunctionOp>(
        loc,
        jacobianFunctionName,
        RealType::get(builder.getContext()),
        mlir::ValueRange(filteredOriginalVariables).getTypes(),
        equation.getNumOfIterationVars(),
        independentVariable.getType().cast<ArrayType>().getRank(),
        RealType::get(builder.getContext()),
        RealType::get(builder.getContext()));

    assert(jacobianFunction.getBodyRegion().empty());
    mlir::Block* bodyBlock = jacobianFunction.addEntryBlock();
    builder.setInsertionPointToStart(bodyBlock);

    // Create the arguments to be passed to the derivative template
    std::vector<mlir::Value> args;

    // 'Time' variable
    args.push_back(jacobianFunction.getTime());

    // The variables
    for (const auto& var : jacobianFunction.getVariables()) {
      if (auto arrayType = var.getType().dyn_cast<ArrayType>(); arrayType && arrayType.isScalar()) {
        args.push_back(builder.create<LoadOp>(var.getLoc(), var));
      } else {
        args.push_back(var);
      }
    }

    // Equation indices
    for (auto equationIndex : jacobianFunction.getEquationIndices()) {
      args.push_back(equationIndex);
    }

    // Seeds for the automatic differentiation
    unsigned int independentVarArgNumber = independentVariable.cast<mlir::BlockArgument>().getArgNumber();
    unsigned int oneSeedPosition = independentVarArgNumber;
    unsigned int alphaSeedPosition = jacobianFunction.getVariables().size();

    if (derivativesMap->hasDerivative(independentVarArgNumber)) {
      alphaSeedPosition = derivativesMap->getDerivative(independentVarArgNumber);
    }

    mlir::Value zero = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 0));
    mlir::Value one = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 1));

    llvm::SmallVector<mlir::Value> seedArrays;

    // Create the seed values for the variables
    for (auto var : llvm::enumerate(mlir::ValueRange(filteredOriginalVariables))) {
      auto varIndex = var.value().cast<mlir::BlockArgument>().getArgNumber();

      if (auto arrayType = var.value().getType().dyn_cast<ArrayType>(); arrayType && !arrayType.isScalar()) {
        assert(arrayType.hasConstantShape());

        auto array = builder.create<AllocOp>(
            loc,
            arrayType.toElementType(RealType::get(builder.getContext())),
            llvm::None);

        seedArrays.push_back(array);
        args.push_back(array);

        builder.create<ArrayFillOp>(loc, array, zero);

        if (varIndex == oneSeedPosition) {
          builder.create<StoreOp>(loc, one, array, jacobianFunction.getVariableIndices());
        } else if (varIndex == alphaSeedPosition) {
          builder.create<StoreOp>(jacobianFunction.getLoc(), jacobianFunction.getAlpha(), array, jacobianFunction.getVariableIndices());
        }
      } else {
        assert(arrayType && arrayType.isScalar());

        if (varIndex == oneSeedPosition) {
          args.push_back(one);
        } else if (varIndex == alphaSeedPosition) {
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

    // Call the derivative template
    auto templateCall = builder.create<CallOp>(
        jacobianFunction.getLoc(), partialDerTemplateName, RealType::get(builder.getContext()), args);

    // Deallocate the seeds
    for (auto seed : seedArrays) {
      builder.create<FreeOp>(jacobianFunction.getLoc(), seed);
    }

    builder.create<mlir::ida::ReturnOp>(jacobianFunction.getLoc(), templateCall.getResult(0));
    return mlir::success();
  }

  mlir::LogicalResult IDASolver::processDeinitFunction(
      mlir::OpBuilder& builder,
      mlir::Value runtimeDataPtr,
      mlir::func::FuncOp deinitFunction)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto terminator = mlir::cast<mlir::func::ReturnOp>(deinitFunction.getBody().back().getTerminator());
    builder.setInsertionPoint(terminator);

    mlir::Value runtimeData = loadRuntimeData(builder, runtimeDataPtr);
    mlir::Value idaInstance = getIDAInstance(builder, runtimeData);

    builder.create<mlir::ida::FreeOp>(idaInstance.getLoc(), idaInstance);

    return mlir::success();
  }

  mlir::LogicalResult IDASolver::processUpdateStatesFunction(
      mlir::OpBuilder& builder,
      mlir::Value runtimeDataPtr,
      mlir::func::FuncOp updateStatesFunction,
      mlir::ValueRange variables)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto terminator = mlir::cast<mlir::func::ReturnOp>(updateStatesFunction.getBody().back().getTerminator());
    builder.setInsertionPoint(terminator);

    mlir::Value runtimeData = loadRuntimeData(builder, runtimeDataPtr);
    mlir::Value idaInstance = getIDAInstance(builder, runtimeData);

    builder.create<mlir::ida::StepOp>(updateStatesFunction.getLoc(), idaInstance);
    return mlir::success();
  }

  bool IDASolver::hasTimeOwnership() const
  {
    return isEnabled();
  }

  mlir::Value IDASolver::getCurrentTime(
      mlir::OpBuilder& builder,
      mlir::Value runtimeDataPtr)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    mlir::Value runtimeData = loadRuntimeData(builder, runtimeDataPtr);
    mlir::Value idaInstance = getIDAInstance(builder, runtimeData);

    return builder.create<mlir::ida::GetCurrentTimeOp>(
        idaInstance.getLoc(), RealType::get(builder.getContext()), idaInstance);
  }

  std::vector<mlir::Value> IDASolver::filterByManagedVariables(mlir::ValueRange variables) const
  {
    std::vector<mlir::Value> result;

    std::vector<mlir::Value> filteredVariables;
    std::vector<mlir::Value> filteredDerivatives;

    for (auto managedVariable : managedVariables) {
      if (derivativesMap->isDerivative(managedVariable.argNumber)) {
        continue;
      }

      filteredVariables.push_back(variables[managedVariable.argNumber]);

      if (derivativesMap->hasDerivative(managedVariable.argNumber)) {
        filteredDerivatives.push_back(variables[derivativesMap->getDerivative(managedVariable.argNumber)]);
      }
    }

    result.insert(result.end(), filteredVariables.begin(), filteredVariables.end());
    result.insert(result.end(), filteredDerivatives.begin(), filteredDerivatives.end());

    return result;
  }
}
