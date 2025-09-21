#include "marco/Codegen/Lowering/StandardFunctionLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering {
StandardFunctionLowerer::StandardFunctionLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

void StandardFunctionLowerer::declare(const ast::StandardFunction &function) {
  mlir::Location location = loc(function.getLocation());

  // Create the record operation.
  auto functionOp = builder().create<FunctionOp>(location, function.getName());

  mlir::OpBuilder::InsertionGuard guard(builder());
  builder().createBlock(&functionOp.getBodyRegion());
  builder().setInsertionPointToStart(functionOp.getBody());

  // Declare the inner classes.
  for (const auto &innerClassNode : function.getInnerClasses()) {
    declare(*innerClassNode->cast<ast::Class>());
  }
}

bool StandardFunctionLowerer::declareVariables(
    const ast::StandardFunction &function) {
  mlir::OpBuilder::InsertionGuard guard(builder());
  LookupScopeGuard lookupScopeGuard(&getContext());

  // Get the operation.
  auto functionOp = mlir::cast<FunctionOp>(getClass(function));
  pushLookupScope(functionOp);
  builder().setInsertionPointToEnd(functionOp.getBody());

  // Declare the variables.
  for (const auto &variable : function.getVariables()) {
    if (!declare(*variable->cast<ast::Member>())) {
      return false;
    }
  }

  // Declare the variables of inner classes.
  for (const auto &innerClassNode : function.getInnerClasses()) {
    if (!declareVariables(*innerClassNode->cast<ast::Class>())) {
      return false;
    }
  }

  return true;
}

bool StandardFunctionLowerer::lower(const ast::StandardFunction &function) {
  mlir::OpBuilder::InsertionGuard guard(builder());

  VariablesSymbolTable::VariablesScope varScope(getVariablesSymbolTable());
  LookupScopeGuard lookupScopeGuard(&getContext());

  // Get the operation.
  auto functionOp = mlir::cast<FunctionOp>(getClass(function));
  pushLookupScope(functionOp);
  builder().setInsertionPointToEnd(functionOp.getBody());

  // Map the variables.
  insertVariable("time", Reference::time(builder(), builder().getUnknownLoc()));

  for (VariableOp variableOp : functionOp.getVariables()) {
    insertVariable(variableOp.getSymName(),
                   Reference::variable(builder(), variableOp->getLoc(),
                                       variableOp.getSymName(),
                                       variableOp.getVariableType().unwrap()));
  }

  // Lower the annotations.
  llvm::SmallVector<llvm::StringRef, 3> inputVarNames;

  for (VariableOp variable : functionOp.getVariables()) {
    if (variable.isInput()) {
      inputVarNames.emplace_back(variable.getName());
    }
  }

  llvm::SmallVector<llvm::StringRef, 1> outputVarNames;

  for (VariableOp variable : functionOp.getVariables()) {
    if (variable.isOutput()) {
      outputVarNames.emplace_back(variable.getName());
    }
  }

  if (function.hasAnnotation()) {
    const auto *annotation = function.getAnnotation();

    // Inline attribute.
    functionOp->setAttr(
        "inline",
        builder().getBoolAttr(function.getAnnotation()->getInlineProperty()));

    // Inverse functions attribute.
    auto inverseFunctionAnnotation = annotation->getInverseFunctionAnnotation();

    InverseFunctionsMap map;

    // Create a map of the function members indexes for faster retrieval.
    llvm::StringMap<unsigned int> indexes;

    for (const auto &name : llvm::enumerate(inputVarNames)) {
      indexes[name.value()] = name.index();
    }

    for (const auto &name : llvm::enumerate(outputVarNames)) {
      indexes[name.value()] = inputVarNames.size() + name.index();
    }

    mlir::StorageUniquer::StorageAllocator allocator;

    // Iterate over the input arguments and for each invertible one
    // add the function to the inverse map.
    for (const auto &arg : inputVarNames) {
      if (!inverseFunctionAnnotation.isInvertible(arg)) {
        continue;
      }

      auto inverseArgs = inverseFunctionAnnotation.getInverseArgs(arg);
      llvm::SmallVector<unsigned int, 3> permutation;

      for (const auto &inverseArg : inverseArgs) {
        assert(indexes.find(inverseArg) != indexes.end());
        permutation.push_back(indexes[inverseArg]);
      }

      map[indexes[arg]] = std::make_pair(
          inverseFunctionAnnotation.getInverseFunction(arg),
          allocator.copyInto(llvm::ArrayRef<unsigned int>(permutation)));
    }

    if (!map.empty()) {
      auto inverseFunctionAttribute =
          InverseFunctionsAttr::get(builder().getContext(), map);

      functionOp->setAttr("inverse", inverseFunctionAttribute);
    }

    if (annotation->hasDerivativeAnnotation()) {
      auto derivativeAnnotation = annotation->getDerivativeAnnotation();

      auto derivativeAttribute = FunctionDerivativeAttr::get(
          builder().getContext(), derivativeAnnotation.getName(),
          derivativeAnnotation.getOrder());

      functionOp.setDerivativeAttr(derivativeAttribute);
    }
  }

  // Create the default values for variables.
  for (const auto &variable : function.getVariables()) {
    if (!lowerVariableDefaultValue(*variable->cast<ast::Member>())) {
      return false;
    }
  }

  // Lower the body.
  if (!lowerClassBody(function)) {
    return false;
  }

  // Create the algorithms.
  for (const auto &algorithm : function.getAlgorithms()) {
    if (!lower(*algorithm->cast<ast::Algorithm>())) {
      return false;
    }
  }

  if (function.isExternal()) {
    if (function.hasExternalFunctionCall()) {
      if (!lowerExternalFunctionCall(function.getExternalLanguage(),
                                     *function.getExternalFunctionCall(),
                                     functionOp)) {
        return false;
      }
    } else {
      if (!createImplicitExternalFunctionCall(function)) {
        return false;
      }
    }
  }

  // Special handling of record constructors.
  if (isRecordConstructor(function)) {
    mlir::Location location = loc(function.getLocation());
    auto algorithmOp = builder().create<AlgorithmOp>(location);

    builder().createBlock(&algorithmOp.getBodyRegion());
    builder().setInsertionPointToStart(algorithmOp.getBody());

    llvm::SmallVector<mlir::Value, 3> args;
    llvm::SmallVector<VariableOp, 1> resultVariables;

    for (VariableOp variableOp : functionOp.getVariables()) {
      if (variableOp.isInput()) {
        args.push_back(builder().create<VariableGetOp>(location, variableOp));
      } else if (variableOp.isOutput()) {
        resultVariables.push_back(variableOp);
      }
    }

    assert(resultVariables.size() == 1);

    mlir::Value record = builder().create<RecordCreateOp>(
        location, resultVariables[0].getVariableType().unwrap(), args);

    builder().create<VariableSetOp>(location, resultVariables[0], record);
    builder().setInsertionPointAfter(algorithmOp);
  }

  // Lower the inner classes.
  for (const auto &innerClassNode : function.getInnerClasses()) {
    if (!lower(*innerClassNode->cast<ast::Class>())) {
      return false;
    }
  }

  return true;
}

bool StandardFunctionLowerer::lowerVariableDefaultValue(
    const ast::Member &variable) {
  if (!variable.hasExpression()) {
    return true;
  }

  const ast::Expression *expression = variable.getExpression();

  mlir::Location expressionLoc = loc(expression->getLocation());

  auto defaultOp =
      builder().create<DefaultOp>(expressionLoc, variable.getName());

  mlir::OpBuilder::InsertionGuard guard(builder());
  mlir::Block *bodyBlock = builder().createBlock(&defaultOp.getBodyRegion());
  builder().setInsertionPointToStart(bodyBlock);

  auto loweredExpression = lower(*expression);
  if (!loweredExpression) {
    return false;
  }
  mlir::Value value = (*loweredExpression)[0].get(expressionLoc);
  builder().create<YieldOp>(expressionLoc, value);
  return true;
}

bool StandardFunctionLowerer::isRecordConstructor(
    const ast::StandardFunction &function) {
  return function.getName().contains("'constructor'");
}

bool StandardFunctionLowerer::lowerExternalFunctionCall(
    llvm::StringRef language,
    const ast::ExternalFunctionCall &externalFunctionCall, FunctionOp funcOp) {
  mlir::SymbolTable symbolTable(funcOp);

  auto algorithmOp =
      builder().create<AlgorithmOp>(loc(externalFunctionCall.getLocation()));

  builder().createBlock(&algorithmOp.getBodyRegion());
  llvm::SmallVector<mlir::Type> expectedResultTypes;

  if (externalFunctionCall.hasDestination()) {
    // Temporarily lower the destination as a read access to determine the type
    // needed for the result of the external function call.
    builder().setInsertionPointToStart(&algorithmOp.getBodyRegion().front());

    auto loweredDestination = lower(*externalFunctionCall.getDestination()
                                         ->cast<ast::ComponentReference>());

    if (!loweredDestination) {
      return false;
    }

    expectedResultTypes.push_back(
        (*loweredDestination)[0]
            .get(loc(externalFunctionCall.getDestination()->getLocation()))
            .getType());

    algorithmOp.getBodyRegion().front().clear();
  }

  // Create the call to the external function.
  builder().setInsertionPointToStart(&algorithmOp.getBodyRegion().front());
  llvm::SmallVector<mlir::Value> callArgs;

  llvm::SmallVector<std::pair<ast::ComponentReference *, mlir::Value>>
      temporaryArrays;

  for (const auto &arg : externalFunctionCall.getArguments()) {
    if (auto componentReference = arg->dyn_cast<ast::ComponentReference>()) {
      // Special handling for protected and output variables, which can be
      // modified by the external function.
      auto variableOp = symbolTable.lookup<VariableOp>(
          componentReference->getElement(0)->getName());

      if (!variableOp) {
        emitIdentifierError(IdentifierError::IdentifierType::VARIABLE,
                            componentReference->getElement(0)->getName(),
                            getVariablesSymbolTable().getVariables(),
                            componentReference->getLocation());
        return false;
      }

      if (!variableOp.isInput() && variableOp.getVariableType().isScalar()) {
        // Extract the previous value and store it into temporary memory.
        auto loweredArg = lower(*arg->cast<ast::Expression>());

        if (!loweredArg) {
          return false;
        }

        mlir::Value currentValue =
            (*loweredArg)[0].get((*loweredArg)[0].getLoc());

        auto allocOp = builder().create<AllocOp>(
            currentValue.getLoc(), ArrayType::get({}, currentValue.getType()),
            mlir::ValueRange());

        builder().create<StoreOp>(currentValue.getLoc(), currentValue, allocOp);
        callArgs.push_back(allocOp);
        temporaryArrays.emplace_back(componentReference, allocOp);
        continue;
      }
    }

    // Read-only argument.
    auto loweredArg = lower(*arg->cast<ast::Expression>());

    if (!loweredArg) {
      return false;
    }

    for (const auto &loweredArgResult : *loweredArg) {
      callArgs.push_back(loweredArgResult.get(loweredArgResult.getLoc()));
    }
  }

  // Create the call to the external function.
  auto callOp = builder().create<ExternalCallOp>(
      loc(externalFunctionCall.getLocation()), expectedResultTypes,
      externalFunctionCall.getCallee(), callArgs);

  callOp.setLanguage(language);

  // Copy the modified scalar arguments back to the original variable.
  for (auto &[componentRef, array] : temporaryArrays) {
    auto loweredDestination = lower(*componentRef);

    if (!loweredDestination) {
      return false;
    }

    mlir::Value loadedValue = builder().create<LoadOp>(array.getLoc(), array);

    if (!this->lowerAssignmentToComponentReference(
            loc(externalFunctionCall.getLocation()), *componentRef,
            loadedValue)) {
      return false;
    }
  }

  // Assign the result of the call to the left-hand side variable.
  if (externalFunctionCall.hasDestination()) {
    return this->lowerAssignmentToComponentReference(
        loc(externalFunctionCall.getLocation()),
        *externalFunctionCall.getDestination()->cast<ast::ComponentReference>(),
        callOp.getResult(0));
  }

  return true;
}

bool StandardFunctionLowerer::createImplicitExternalFunctionCall(
    const ast::Function &function) {
  auto algorithmOp = builder().create<AlgorithmOp>(loc(function.getLocation()));

  builder().createBlock(&algorithmOp.getBodyRegion());
  llvm::SmallVector<mlir::Type> expectedResultTypes;

  // Temporarily lower the destination as a read access to determine the type
  // needed for the result of the external function call.
  builder().setInsertionPointToStart(&algorithmOp.getBodyRegion().front());

  for (const auto &variable : function.getVariables()) {
    if (variable->cast<ast::Member>()->isOutput()) {
      if (!expectedResultTypes.empty()) {
        // Multiple output variables are not supported for implicit external
        // function calls.
        emitError(loc(function.getLocation()))
            << "Functions with multiple output variables are not supported "
               "for implicit external function calls";

        return false;
      }

      expectedResultTypes.push_back(
          lookupVariable(variable->cast<ast::Member>()->getName())
              ->get(loc(variable->getLocation()))
              .getType());
    }
  }

  algorithmOp.getBodyRegion().front().clear();

  // Create the call to the external function.
  builder().setInsertionPointToStart(&algorithmOp.getBodyRegion().front());
  llvm::SmallVector<mlir::Value> callArgs;

  for (const auto &variable : function.getVariables()) {
    if (variable->cast<ast::Member>()->isOutput()) {
      continue;
    }

    auto variableRef = lookupVariable(variable->cast<ast::Member>()->getName());
    callArgs.push_back(variableRef->get(variableRef->getLoc()));
  }

  auto callOp = builder().create<ExternalCallOp>(loc(function.getLocation()),
                                                 expectedResultTypes,
                                                 function.getName(), callArgs);

  callOp.setLanguage(function.getExternalLanguage());
  ;

  for (const auto &variable : function.getVariables()) {
    if (variable->cast<ast::Member>()->isOutput()) {
      lookupVariable(variable->cast<ast::Member>()->getName())
          ->set(loc(variable->getLocation()), {}, callOp.getResult(0));
    }
  }

  return true;
}
} // namespace marco::codegen::lowering
