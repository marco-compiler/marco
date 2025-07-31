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

  if (function.hasExternalRef() && function.getExternalRef()->hasExternalFunctionCall()){

    llvm::SmallVector<mlir::Type> inputTypes;
    llvm::SmallVector<mlir::Type> outputTypes;

    for (const auto &variableNode : function.getVariables()) {
      const auto* member = variableNode->cast<ast::Member>();
      if (member->isOutput()) {
        const ast::VariableType* astVariableType = member->getType();
        const ast::TypePrefix* astTypePrefix = member->getTypePrefix();  
        std::optional<VariableType> mlirVariableType = getVariableType(*astVariableType, *astTypePrefix);
        outputTypes.push_back(mlirVariableType->unwrap());
      }
    }

    auto args = function.getExternalRef()->getExternalFunctionCall()->getExpressions(); 

    for (size_t i = 0; i < args.size(); i++){
      auto argType = lowerArg(*args[i]->cast<ast::Expression>());
      inputTypes.push_back(argType); 
    }
    
    mlir::FunctionType funcType = mlir::FunctionType::get(
      builder().getContext(), 
      inputTypes,             
      outputTypes         
    );

    ExternalFunctionOp externalFunctionOp; 
    
    {
      auto module = functionOp->getParentOfType<mlir::ModuleOp>();
      mlir::OpBuilder::InsertionGuard guard(builder());
      builder().setInsertionPointToStart(module.getBody());
      externalFunctionOp = builder().create<ExternalFunctionOp>(loc(function.getExternalRef()->getExternalFunctionCall()->getLocation()), function.getExternalRef()->getExternalFunctionCall()->getName(), funcType);
      externalFunctionOp.setPrivate();
    }

  }

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

  if (function.hasExternalRef() && function.getExternalRef()->hasExternalFunctionCall()){

    mlir::Location location = loc(function.getLocation());

    auto algorithmOp = builder().create<AlgorithmOp>(location);

    mlir::OpBuilder::InsertionGuard bodyGuard(builder()); 
    builder().createBlock(&algorithmOp.getBodyRegion());
    builder().setInsertionPointToStart(algorithmOp.getBody());
    
    lower(*(function.getExternalRef()->getExternalFunctionCall()));
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

std::optional<VariableType>
StandardFunctionLowerer::getVariableType(const ast::VariableType &variableType,
                              const ast::TypePrefix &typePrefix) {
  llvm::SmallVector<int64_t, 3> shape;

  for (size_t i = 0, rank = variableType.getRank(); i < rank; ++i) {
    const ast::ArrayDimension *dimension = variableType[i];

    if (dimension->isDynamic()) {
      shape.push_back(VariableType::kDynamic);
    } else {
      shape.push_back(dimension->getNumericSize());
    }
  }

  mlir::Type baseType;

  if (auto builtInType = variableType.dyn_cast<ast::BuiltInType>()) {
    if (builtInType->getBuiltInTypeKind() == ast::BuiltInType::Kind::Boolean) {
      baseType = BooleanType::get(builder().getContext());
    } else if (builtInType->getBuiltInTypeKind() ==
               ast::BuiltInType::Kind::Integer) {
      baseType = IntegerType::get(builder().getContext());
    } else if (builtInType->getBuiltInTypeKind() ==
               ast::BuiltInType::Kind::Real) {
      baseType = RealType::get(builder().getContext());
    } else {
      llvm_unreachable("Unknown built-in type");
      return nullptr;
    }
  } else if (auto userDefinedType =
                 variableType.dyn_cast<ast::UserDefinedType>()) {
    auto symbolOp = resolveType(*userDefinedType, getLookupScope());

    if (!symbolOp) {
      return std::nullopt;
    }

    if (mlir::isa<RecordOp>(*symbolOp)) {
      baseType = RecordType::get(builder().getContext(),
                                 getSymbolRefFromRoot(*symbolOp));
    } else {
      llvm_unreachable("Unknown variable type");
      return nullptr;
    }
  } else {
    llvm_unreachable("Unknown variable type");
    return nullptr;
  }

  VariabilityProperty variabilityProperty = VariabilityProperty::none;
  IOProperty ioProperty = IOProperty::none;

  if (typePrefix.isDiscrete()) {
    variabilityProperty = VariabilityProperty::discrete;
  } else if (typePrefix.isParameter()) {
    variabilityProperty = VariabilityProperty::parameter;
  } else if (typePrefix.isConstant()) {
    variabilityProperty = VariabilityProperty::constant;
  }

  if (typePrefix.isInput()) {
    ioProperty = IOProperty::input;
  } else if (typePrefix.isOutput()) {
    ioProperty = IOProperty::output;
  }

  return VariableType::get(shape, baseType, variabilityProperty, ioProperty);
}

mlir::Type StandardFunctionLowerer::lowerArg(const ast::Expression &expression) {
  mlir::Location location = loc(expression.getLocation());
  auto loweredExpression = lower(expression);
  auto &results = *loweredExpression;
  assert(results.size() == 1);
  return results[0].get(location).getType();
}

} // namespace marco::codegen::lowering
