#include "marco/Codegen/Transforms/ModelSolving/ExternalSolvers/IDASolver.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "marco/Codegen/Utils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <queue>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  IDASolver::IDASolver(
      mlir::TypeConverter* typeConverter,
      const mlir::BlockAndValueMapping& derivatives,
      IDAOptions options,
      double startTime,
      double endTime,
      double timeStep)
    : ExternalSolver(typeConverter),
      derivatives(&derivatives),
      enabled(true),
      options(std::move(options)),
      startTime(startTime),
      endTime(endTime),
      timeStep(timeStep)
  {
    assert(startTime < endTime);
    assert(timeStep > 0);
    assert(options.relativeTolerance > 0);
    assert(options.absoluteTolerance > 0);
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
    structTypes.push_back(getTypeConverter()->convertType(mlir::ida::InstanceType::get(context)));

    for (size_t i = 0; i < managedVariables.size(); ++i) {
      structTypes.push_back(getTypeConverter()->convertType(mlir::ida::VariableType::get(context)));
    }

    return mlir::LLVM::LLVMStructType::getLiteral(context, structTypes);
  }

  bool IDASolver::hasVariable(mlir::Value variable) const
  {
    return llvm::find(managedVariables, variable) != managedVariables.end();
  }

  void IDASolver::addVariable(mlir::Value variable)
  {
    assert(variable.isa<mlir::BlockArgument>());

    if (!hasVariable(variable)) {
      managedVariables.push_back(variable);
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
      mlir::FuncOp initFunction,
      mlir::ValueRange variables,
      const Model<ScheduledEquationsBlock>& model)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto terminator = mlir::cast<mlir::ReturnOp>(initFunction.getBody().back().getTerminator());
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
    builder.create<mlir::ida::SetRelativeToleranceOp>(idaInstance.getLoc(), idaInstance, builder.getF64FloatAttr(options.relativeTolerance));
    builder.create<mlir::ida::SetAbsoluteToleranceOp>(idaInstance.getLoc(), idaInstance, builder.getF64FloatAttr(options.absoluteTolerance));

    // Store the IDA instance into the runtime data structure
    mlir::Value runtimeData = loadRuntimeData(builder, runtimeDataPtr);
    runtimeData = setIDAInstance(builder, runtimeData, idaInstance);
    storeRuntimeData(builder, runtimeDataPtr, runtimeData);

    // Add the variables to IDA
    if (auto res = addVariablesToIDA(builder, runtimeDataPtr, variables); mlir::failed(res)) {
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
      mlir::Value runtimeDataPtr,
      mlir::ValueRange variables)
  {
    mlir::Value runtimeData = loadRuntimeData(builder, runtimeDataPtr);
    mlir::Value idaInstance = getIDAInstance(builder, runtimeData);

    mlir::BlockAndValueMapping inverseDerivatives = derivatives->getInverse();

    // Declare the variables and their derivatives inside IDA
    unsigned int idaVariableIndex = 0;

    for (const auto& variable : managedVariables) {
      auto variableIndex = variable.cast<mlir::BlockArgument>().getArgNumber();

      auto arrayType = variable.getType().cast<ArrayType>();
      assert(arrayType.hasConstantShape());

      if (derivatives->contains(variable)) {
        // State variable
        mlir::Value stateVariable = variables[variableIndex];

        mlir::Value derivative = derivatives->lookup(variable);
        auto derivativeIndex = derivative.cast<mlir::BlockArgument>().getArgNumber();
        derivative = variables[derivativeIndex];

        mlir::Value idaVariable = builder.create<mlir::ida::AddStateVariableOp>(
            variable.getLoc(),
            idaInstance,
            stateVariable,
            derivative,
            builder.getI64ArrayAttr(arrayType.getShape()));

        runtimeData = setIDAVariable(builder, runtimeData, idaVariableIndex, idaVariable);

        mappedVariables[variableIndex] = idaVariableIndex;
        mappedVariables[derivativeIndex] = idaVariableIndex;

        ++idaVariableIndex;

      } else if (!inverseDerivatives.contains(variable)) {
        // Algebraic variable (that is, the variable is neither a state
        // nor a derivative).
        mlir::Value algebraicVariable = variables[variableIndex];

        auto idaVariable = builder.create<mlir::ida::AddAlgebraicVariableOp>(
            variable.getLoc(),
            idaInstance,
            algebraicVariable,
            builder.getI64ArrayAttr(arrayType.getShape()));

        runtimeData = setIDAVariable(builder, runtimeData, idaVariableIndex, idaVariable);
        mappedVariables[variableIndex] = idaVariableIndex;

        ++idaVariableIndex;
      }
    }

    storeRuntimeData(builder, runtimeDataPtr, runtimeData);

    // Initialize the variables owned by IDA with the same variables held by MARCO
    for (const auto& variable : managedVariables) {
      auto variableIndex = variable.cast<mlir::BlockArgument>().getArgNumber();
      assert(variableIndex < variables.size());
      mlir::Value source = variables[variableIndex];
      assert(variable.getType() == source.getType());

      auto arrayType = source.getType().cast<ArrayType>();
      assert(arrayType.hasConstantShape());

      mlir::Value idaVariable = getIDAVariable(builder, runtimeData, mappedVariables[variableIndex]);

      if (inverseDerivatives.contains(variable)) {
        // If the variable is a derivative
        auto castedVariable = builder.create<mlir::ida::GetDerivativeOp>(
            idaVariable.getLoc(), arrayType, idaInstance, idaVariable);

        // Fill with source
        copyArray(builder, idaVariable.getLoc(), source, castedVariable);
      } else {
        auto castedVariable = builder.create<mlir::ida::GetVariableOp>(
            idaVariable.getLoc(), arrayType, idaInstance, idaVariable);

        // Fill with source
        copyArray(builder, idaVariable.getLoc(), source, castedVariable);
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult IDASolver::addEquationsToIDA(
      mlir::OpBuilder& builder,
      mlir::Value runtimeDataPtr,
      const Model<ScheduledEquationsBlock>& model)
  {
    mlir::Value runtimeData = loadRuntimeData(builder, runtimeDataPtr);
    mlir::Value idaInstance = getIDAInstance(builder, runtimeData);

    // Substitute the accesses to non-IDA variables with the equations writing in such variables
    std::vector<std::unique_ptr<ScheduledEquation>> independentEquations;
    std::multimap<unsigned int, std::pair<modeling::MultidimensionalRange, ScheduledEquation*>> writesMap;

    for (const auto& equationsBlock : model.getScheduledBlocks()) {
      for (const auto& equation : *equationsBlock) {
        if (equations.find(equation.get()) == equations.end()) {
          const auto& write = equation->getWrite();
          auto varPosition = write.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
          auto writtenIndices = write.getAccessFunction().map(equation->getIterationRanges());
          writesMap.emplace(varPosition, std::make_pair(writtenIndices, equation.get()));
        }
      }
    }

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
          // Avoid unnecessary duplicates
          break;
        }

        auto readIndices = access.getAccessFunction().map(equation->getIterationRanges());
        auto varPosition = access.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
        auto writingEquations = llvm::make_range(writesMap.equal_range(varPosition));

        for (const auto& entry : writingEquations) {
          ScheduledEquation* writingEquation = entry.second.second;
          auto writtenVariableIndices = entry.second.first;

          if (!writtenVariableIndices.overlaps(readIndices)) {
            continue;
          }

          atLeastOneAccessReplaced = true;

          auto clone = Equation::build(equation->cloneIR(), equation->getVariables());

          auto explicitWritingEquation = writingEquation->cloneIRAndExplicitate(builder);
          TemporaryEquationGuard guard(*explicitWritingEquation);
          auto res = explicitWritingEquation->replaceInto(builder, *clone, access.getAccessFunction(), access.getPath());
          assert(mlir::succeeded(res));

          // Add the equation with the replaced access
          auto readAccessIndices = access.getAccessFunction().inverseMap(
              modeling::IndexSet(writtenVariableIndices),
              modeling::IndexSet(equation->getIterationRanges()));

          auto newEquationIndices = readAccessIndices.intersect(equation->getIterationRanges());

          for (const auto& range : newEquationIndices) {
            auto matchedEquation = std::make_unique<MatchedEquation>(
                clone->clone(), range, equation->getWrite().getPath());

            auto scheduledEquation = std::make_unique<ScheduledEquation>(
                std::move(matchedEquation), range, equation->getSchedulingDirection());

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

    mlir::BlockAndValueMapping inverseDerivatives = derivatives->getInverse();
    auto equationVariables = model.getOperation().bodyRegion().getArguments();

    size_t residualFunctionsCounter = 0;
    size_t jacobianFunctionsCounter = 0;
    size_t partialDerTemplatesCounter = 0;

    for (const auto& equation : independentEquations) {
      auto ranges = equation->getIterationRanges();
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
      auto residualFunctionName = "residualFunction" + std::to_string(residualFunctionsCounter++);

      if (auto res = createResidualFunction(builder, *equation, equationVariables, idaEquation, residualFunctionName); mlir::failed(res)) {
        return res;
      }

      builder.create<mlir::ida::AddResidualOp>(
          equation->getOperation().getLoc(), idaInstance, idaEquation, residualFunctionName);

      // Create the partial derivative template
      std::string partialDerTemplateName = "ida_pder_" + std::to_string(partialDerTemplatesCounter++);

      if (auto res = createPartialDerTemplateFunction(builder, *equation, equationVariables, partialDerTemplateName); mlir::failed(res)) {
        return res;
      }

      // Create the Jacobian functions
      for (const auto& variable : managedVariables) {
        auto variableIndex = variable.cast<mlir::BlockArgument>().getArgNumber();

        if (inverseDerivatives.contains(variable)) {
          // If the variable is a derivative, then skip the creation of the Jacobian functions
          // because it is already done when encountering the state variable.
          continue;
        }

        auto jacobianFunctionName = "jacobianFunction" + std::to_string(jacobianFunctionsCounter++);

        if (auto res = createJacobianFunction(builder, *equation, equationVariables, jacobianFunctionName, variable, partialDerTemplateName); mlir::failed(res)) {
          return res;
        }

        builder.create<mlir::ida::AddJacobianOp>(
            equation->getOperation().getLoc(),
            idaInstance,
            idaEquation,
            getIDAVariable(builder, runtimeData, mappedVariables[variableIndex]),
            jacobianFunctionName);
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
      auto accessedVariableIndex = access.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
      mlir::Value idaVariable = getIDAVariable(builder, runtimeData, mappedVariables[accessedVariableIndex]);

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

    assert(residualFunction.bodyRegion().empty());
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

    auto clonedTerminator = mlir::cast<EquationSidesOp>(residualFunction.bodyRegion().back().getTerminator());

    assert(clonedTerminator.lhsValues().size() == 1);
    assert(clonedTerminator.rhsValues().size() == 1);

    mlir::Value lhs = clonedTerminator.lhsValues()[0];
    mlir::Value rhs = clonedTerminator.rhsValues()[0];

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

    auto lhsOp = clonedTerminator.lhs().getDefiningOp<EquationSideOp>();
    auto rhsOp = clonedTerminator.rhs().getDefiningOp<EquationSideOp>();
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

    for (auto type : partialDerTemplate.getType().getInputs()) {
      args.push_back(type);
    }

    partialDerTemplate->setAttr(
        partialDerTemplate.typeAttrName(),
        mlir::TypeAttr::get(builder.getFunctionType(args, partialDerTemplate.getType().getResults())));

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
      if (auto arrayType = type.dyn_cast<ArrayType>()) {
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
    mlir::Block* entryBlock = builder.createBlock(&functionOp.body());
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
    assert(originalTerminator.lhsValues().size() == 1);
    assert(originalTerminator.rhsValues().size() == 1);

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
        // We need to check if the load operation is performed on model variable.
        // Those variables are in fact created inside the function by means of MemberCreateOps,
        // and if such variable is a scalar one there would be a load operation wrongly operating
        // on a scalar value.

        if (auto memberLoadOp = mapping.lookup(loadOp.array()).getDefiningOp<MemberLoadOp>()) {
          mapping.map(loadOp.getResult(), memberLoadOp.getResult());
          continue;
        }
      }

      builder.clone(op, mapping);
    }

    auto terminator = mlir::cast<EquationSidesOp>(functionOp.bodyBlock()->getTerminator());
    assert(terminator.lhsValues().size() == 1);
    assert(terminator.rhsValues().size() == 1);

    mlir::Value lhs = terminator.lhsValues()[0];
    mlir::Value rhs = terminator.rhsValues()[0];

    if (auto arrayType = lhs.getType().dyn_cast<ArrayType>()) {
      assert(rhs.getType().isa<ArrayType>());
      assert(arrayType.getRank() + explicitEquationInductions.size() == inductions.size());
      auto implicitInductions = llvm::makeArrayRef(inductions).take_back(arrayType.getRank());

      lhs = builder.create<LoadOp>(loc, lhs, implicitInductions);
      rhs = builder.create<LoadOp>(loc, rhs, implicitInductions);
    }

    auto result = builder.create<SubOp>(loc, RealType::get(builder.getContext()), rhs, lhs);
    builder.create<MemberStoreOp>(loc, outputMember, result);

    auto lhsOp = terminator.lhs().getDefiningOp<EquationSideOp>();
    auto rhsOp = terminator.rhs().getDefiningOp<EquationSideOp>();
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
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(equation.getOperation()->getParentOfType<mlir::ModuleOp>().getBody());

    // Pass only the variables that are managed by IDA
    auto filteredOriginalVariables = filterByManagedVariables(equationVariables);

    auto jacobianFunction = builder.create<mlir::ida::JacobianFunctionOp>(
        equation.getOperation().getLoc(),
        jacobianFunctionName,
        RealType::get(builder.getContext()),
        mlir::ValueRange(filteredOriginalVariables).getTypes(),
        equation.getNumOfIterationVars(),
        independentVariable.getType().cast<ArrayType>().getRank(),
        RealType::get(builder.getContext()),
        RealType::get(builder.getContext()));

    assert(jacobianFunction.bodyRegion().empty());
    mlir::Block* bodyBlock = jacobianFunction.addEntryBlock();

    // Create the call to the derivative template
    builder.setInsertionPointToStart(bodyBlock);

    std::vector<mlir::Value> args;

    args.push_back(jacobianFunction.getTime());

    for (const auto& var : jacobianFunction.getVariables()) {
      if (auto arrayType = var.getType().dyn_cast<ArrayType>(); arrayType && arrayType.isScalar()) {
        args.push_back(builder.create<LoadOp>(var.getLoc(), var));
      } else {
        args.push_back(var);
      }
    }

    for (auto equationIndex : jacobianFunction.getEquationIndices()) {
      args.push_back(equationIndex);
    }

    unsigned int oneSeedPosition = independentVariable.cast<mlir::BlockArgument>().getArgNumber();
    unsigned int alphaSeedPosition = jacobianFunction.getVariables().size();

    if (derivatives->contains(independentVariable)) {
      alphaSeedPosition = derivatives->lookup(independentVariable).cast<mlir::BlockArgument>().getArgNumber();
    }

    mlir::Value zero = builder.create<ConstantOp>(jacobianFunction.getLoc(), RealAttr::get(builder.getContext(), 0));
    mlir::Value one = builder.create<ConstantOp>(jacobianFunction.getLoc(), RealAttr::get(builder.getContext(), 1));

    // Create the seed values for the variables
    for (auto varType : llvm::enumerate(mlir::ValueRange(filteredOriginalVariables).getTypes())) {
      if (auto arrayType = varType.value().dyn_cast<ArrayType>(); arrayType && !arrayType.isScalar()) {
        assert(arrayType.hasConstantShape());

        auto array = builder.create<AllocOp>(
            jacobianFunction.getLoc(),
            arrayType.toElementType(RealType::get(builder.getContext())),
            llvm::None);

        args.push_back(array);

        builder.create<ArrayFillOp>(jacobianFunction.getLoc(), array, zero);

        if (varType.index() == oneSeedPosition) {
          builder.create<StoreOp>(jacobianFunction.getLoc(), one, array, jacobianFunction.getVariableIndices());
        } else if (varType.index() == alphaSeedPosition) {
          builder.create<StoreOp>(jacobianFunction.getLoc(), jacobianFunction.getAlpha(), array, jacobianFunction.getVariableIndices());
        }

      } else {
        assert(arrayType && arrayType.isScalar());

        if (varType.index() == oneSeedPosition) {
          args.push_back(one);
        } else if (varType.index() == alphaSeedPosition) {
          args.push_back(jacobianFunction.getAlpha());
        } else {
          args.push_back(zero);
        }
      }
    }

    for (size_t i = 0; i < jacobianFunction.getEquationIndices().size(); ++i) {
      // The indices of the equation have a seed equal to zero
      args.push_back(zero);
    }

    auto templateCall = builder.create<CallOp>(
        jacobianFunction.getLoc(), partialDerTemplateName, RealType::get(builder.getContext()), args);

    builder.create<mlir::ida::ReturnOp>(jacobianFunction.getLoc(), templateCall.getResult(0));

    return mlir::success();
  }

  mlir::LogicalResult IDASolver::processDeinitFunction(
      mlir::OpBuilder& builder,
      mlir::Value runtimeDataPtr,
      mlir::FuncOp deinitFunction)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto terminator = mlir::cast<mlir::ReturnOp>(deinitFunction.body().back().getTerminator());
    builder.setInsertionPoint(terminator);

    mlir::Value runtimeData = loadRuntimeData(builder, runtimeDataPtr);
    mlir::Value idaInstance = getIDAInstance(builder, runtimeData);

    builder.create<mlir::ida::FreeOp>(idaInstance.getLoc(), idaInstance);

    return mlir::success();
  }

  mlir::LogicalResult IDASolver::processUpdateStatesFunction(
      mlir::OpBuilder& builder,
      mlir::Value runtimeDataPtr,
      mlir::FuncOp updateStatesFunction,
      mlir::ValueRange variables)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto terminator = mlir::cast<mlir::ReturnOp>(updateStatesFunction.body().back().getTerminator());
    builder.setInsertionPoint(terminator);

    mlir::Value runtimeData = loadRuntimeData(builder, runtimeDataPtr);
    mlir::Value idaInstance = getIDAInstance(builder, runtimeData);

    if (options.equidistantTimeGrid) {
      builder.create<mlir::ida::StepOp>(
          updateStatesFunction.getLoc(),
          idaInstance,
          builder.getF64FloatAttr(timeStep));
    } else {
      builder.create<mlir::ida::StepOp>(updateStatesFunction.getLoc(), idaInstance);
    }

    // Copy back the values from IDA to MARCO
    mlir::BlockAndValueMapping inverseDerivatives = derivatives->getInverse();

    for (const auto& variable : managedVariables) {
      auto variableIndex = variable.cast<mlir::BlockArgument>().getArgNumber();
      assert(variableIndex < variables.size());
      mlir::Value destination = variables[variableIndex];
      assert(variable.getType() == destination.getType());
      auto arrayType = variable.getType().cast<ArrayType>();

      mlir::Value idaVariable = getIDAVariable(builder, runtimeData, mappedVariables[variableIndex]);

      if (inverseDerivatives.contains(variable)) {
        auto source = builder.create<mlir::ida::GetDerivativeOp>(
            idaVariable.getLoc(), arrayType, idaInstance, idaVariable);

        // Fill with source
        copyArray(builder, idaVariable.getLoc(), source, destination);
      } else {
        auto source = builder.create<mlir::ida::GetVariableOp>(
            idaVariable.getLoc(), arrayType, idaInstance, idaVariable);

        // Fill with source
        copyArray(builder, idaVariable.getLoc(), source, destination);
      }
    }

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

    mlir::BlockAndValueMapping inverseDerivatives = derivatives->getInverse();

    for (auto variable : managedVariables) {
      if (inverseDerivatives.contains(variable)) {
        continue;
      }

      auto variableIndex = variable.cast<mlir::BlockArgument>().getArgNumber();
      filteredVariables.push_back(variables[variableIndex]);

      if (auto derivative = derivatives->lookupOrNull(variable)) {
        auto derivativeIndex = derivative.cast<mlir::BlockArgument>().getArgNumber();
        filteredDerivatives.push_back(variables[derivativeIndex]);
      }
    }

    result.insert(result.end(), filteredVariables.begin(), filteredVariables.end());
    result.insert(result.end(), filteredDerivatives.begin(), filteredDerivatives.end());

    return result;
  }
}
