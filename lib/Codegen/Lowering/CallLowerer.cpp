#include "marco/Codegen/Lowering/CallLowerer.h"
#include "llvm/ADT/StringSwitch.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  CallLowerer::CallLowerer(BridgeInterface* bridge)
    : Lowerer(bridge)
  {
  }

  Results CallLowerer::lower(const ast::Call& call)
  {
    const ast::ComponentReference* callee =
        call.getCallee()->cast<ast::ComponentReference>();

    std::optional<mlir::Operation*> calleeOp = resolveCallee(*callee);

    if (!calleeOp) {
      llvm_unreachable("Invalid callee");
      return {};
    }

    if (*calleeOp) {
      if (mlir::isa<FunctionOp, DerFunctionOp>(*calleeOp)) {
        // User-defined function.
        llvm::SmallVector<VariableOp> inputVariables;

        if (auto functionOp = mlir::dyn_cast<FunctionOp>(*calleeOp)) {
          getCustomFunctionInputVariables(inputVariables, functionOp);
        }

        if (auto derFunctionOp = mlir::dyn_cast<DerFunctionOp>(*calleeOp)) {
          getCustomFunctionInputVariables(inputVariables, derFunctionOp);
        }

        llvm::SmallVector<std::string, 3> argNames;
        llvm::SmallVector<mlir::Value, 3> argValues;

        lowerCustomFunctionArgs(call, inputVariables, argNames, argValues);
        assert(argNames.empty() && "Named arguments not supported yet");

        llvm::SmallVector<int64_t, 3> expectedArgRanks;
        getFunctionExpectedArgRanks(*calleeOp, expectedArgRanks);

        llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
        getFunctionResultTypes(*calleeOp, scalarizedResultTypes);

        llvm::SmallVector<mlir::Type, 1> resultTypes;

        if (!getVectorizedResultTypes(
                argValues, expectedArgRanks,
                scalarizedResultTypes, resultTypes)) {
          assert("Can't vectorize function call");
          return {};
        }

        auto callOp = builder().create<CallOp>(
            loc(call.getLocation()),
            getSymbolRefFromRoot(*calleeOp),
            resultTypes, argValues);

        std::vector<Reference> results;

        for (auto result : callOp->getResults()) {
          results.push_back(Reference::ssa(builder(), result));
        }

        return Results(results.begin(), results.end());
      }

      // Check if it's an implicit record constructor.
      if (auto recordConstructor = mlir::dyn_cast<RecordOp>(*calleeOp)) {
        llvm::SmallVector<VariableOp> inputVariables;
        getRecordConstructorInputVariables(inputVariables, recordConstructor);

        llvm::SmallVector<std::string, 3> argNames;
        llvm::SmallVector<mlir::Value, 3> argValues;
        lowerRecordConstructorArgs(call, inputVariables, argNames, argValues);
        assert(argNames.empty() && "Named args for records not yet supported");

        mlir::SymbolRefAttr symbol = getSymbolRefFromRoot(recordConstructor);

        mlir::Value result = builder().create<RecordCreateOp>(
            loc(call.getLocation()),
            RecordType::get(builder().getContext(), symbol),
            argValues);

        return Reference::ssa(builder(), result);
      }
    }

    if (isBuiltInFunction(*callee)) {
      // Built-in function.
      return dispatchBuiltInFunctionCall(call);
    }

    // The function doesn't exist.
    llvm_unreachable("Function not found");
    return {};
  }

  std::optional<mlir::Operation*> CallLowerer::resolveCallee(
      const ast::ComponentReference& callee)
  {
    size_t pathLength = callee.getPathLength();
    assert(callee.getPathLength() > 0);

    for (size_t i = 0; i < pathLength; ++i) {
      if (callee.getElement(i)->getNumOfSubscripts() != 0) {
        return std::nullopt;
      }
    }

    mlir::Operation* result = resolveSymbolName(
        callee.getElement(0)->getName(), getLookupScope());

    for (size_t i = 1; i < pathLength; ++i) {
      if (result == nullptr) {
        return nullptr;
      }

      result = getSymbolTable().lookupSymbolIn(
          result, builder().getStringAttr(callee.getElement(i)->getName()));
    }

    return result;
  }

  mlir::Value CallLowerer::lowerArg(const ast::Expression& expression)
  {
    mlir::Location location = loc(expression.getLocation());
    auto results = lower(expression);
    assert(results.size() == 1);
    return results[0].get(location);
  }

  void CallLowerer::getCustomFunctionInputVariables(
      llvm::SmallVectorImpl<mlir::modelica::VariableOp>& inputVariables,
      FunctionOp functionOp)
  {
    for (VariableOp variableOp : functionOp.getVariables()) {
      if (variableOp.isInput()) {
        inputVariables.push_back(variableOp);
      }
    }
  }

  void CallLowerer::getCustomFunctionInputVariables(
      llvm::SmallVectorImpl<mlir::modelica::VariableOp>& inputVariables,
      DerFunctionOp derFunctionOp)
  {
    mlir::Operation* derivedFunctionOp = derFunctionOp.getOperation();

    while (derivedFunctionOp && !mlir::isa<FunctionOp>(derivedFunctionOp)) {
      derivedFunctionOp = resolveSymbolName<FunctionOp, DerFunctionOp>(
          derFunctionOp.getDerivedFunction(),
          derivedFunctionOp);
    }

    assert(derivedFunctionOp && "Derived function not found");
    auto functionOp = mlir::cast<FunctionOp>(derivedFunctionOp);
    getCustomFunctionInputVariables(inputVariables, functionOp);
  }

  void CallLowerer::lowerCustomFunctionArgs(
      const ast::Call& call,
      llvm::ArrayRef<VariableOp> calleeInputs,
      llvm::SmallVectorImpl<std::string>& argNames,
      llvm::SmallVectorImpl<mlir::Value>& argValues)
  {
    size_t numOfArgs = call.getNumOfArguments();

    if (numOfArgs != 0) {
      if (auto reductionArg =
              call.getArgument(0)
                  ->dyn_cast<ast::ReductionFunctionArgument>()) {
        assert(call.getNumOfArguments() == 1);
        llvm_unreachable("ReductionOp has not been implemented yet");
        return;
      }
    }

    bool existsNamedArgument = false;

    for (size_t i = 0; i < numOfArgs && !existsNamedArgument; ++i) {
      if (call.getArgument(i)->isa<ast::NamedFunctionArgument>()) {
        existsNamedArgument = true;
      }
    }

    size_t argIndex = 0;

    // Process the unnamed arguments.
    while (argIndex < numOfArgs &&
           !call.getArgument(argIndex)->isa<ast::NamedFunctionArgument>()) {
      auto arg = call.getArgument(argIndex)
                     ->cast<ast::ExpressionFunctionArgument>();

      argValues.push_back(lowerArg(*arg->getExpression()));

      if (existsNamedArgument) {
        VariableOp variableOp = calleeInputs[argIndex];
        argNames.push_back(variableOp.getSymName().str());
      }

      ++argIndex;
    }

    // Process the named arguments.
    while (argIndex < numOfArgs) {
      auto arg = call.getArgument(argIndex)
                     ->cast<ast::NamedFunctionArgument>();

      argValues.push_back(lowerArg(
          *arg->getValue()
               ->cast<ast::ExpressionFunctionArgument>()
               ->getExpression()));

      argNames.push_back(arg->getName().str());
      ++argIndex;
    }
  }

  void CallLowerer::getRecordConstructorInputVariables(
      llvm::SmallVectorImpl<mlir::modelica::VariableOp>& inputVariables,
      mlir::modelica::RecordOp recordOp)
  {
    for (VariableOp variableOp : recordOp.getVariables()) {
      if (variableOp.isInput()) {
        inputVariables.push_back(variableOp);
      }
    }
  }

  void CallLowerer::lowerRecordConstructorArgs(
      const ast::Call& call,
      llvm::ArrayRef<mlir::modelica::VariableOp> calleeInputs,
      llvm::SmallVectorImpl<std::string>& argNames,
      llvm::SmallVectorImpl<mlir::Value>& argValues)
  {
    assert(llvm::none_of(call.getArguments(), [](const auto& arg) {
      return arg->template isa<ast::ReductionFunctionArgument>();
    }));

    size_t numOfArgs = call.getNumOfArguments();
    bool existsNamedArgument = false;

    for (size_t i = 0; i < numOfArgs && !existsNamedArgument; ++i) {
      if (call.getArgument(i)->isa<ast::NamedFunctionArgument>()) {
        existsNamedArgument = true;
      }
    }

    size_t argIndex = 0;

    // Process the unnamed arguments.
    while (argIndex < numOfArgs &&
           !call.getArgument(argIndex)->isa<ast::NamedFunctionArgument>()) {
      auto arg = call.getArgument(argIndex)
                     ->cast<ast::ExpressionFunctionArgument>();

      argValues.push_back(lowerArg(*arg->getExpression()));

      if (existsNamedArgument) {
        VariableOp variableOp = calleeInputs[argIndex];
        argNames.push_back(variableOp.getSymName().str());
      }

      ++argIndex;
    }

    // Process the named arguments.
    while (argIndex < numOfArgs) {
      auto arg = call.getArgument(argIndex)
                     ->cast<ast::NamedFunctionArgument>();

      argValues.push_back(lowerArg(
          *arg->getValue()
               ->cast<ast::ExpressionFunctionArgument>()
                   ->getExpression()));

      argNames.push_back(arg->getName().str());
      ++argIndex;
    }
  }

  void CallLowerer::lowerBuiltInFunctionArgs(
      const ast::Call& call,
      llvm::SmallVectorImpl<mlir::Value>& args)
  {
    assert(llvm::none_of(call.getArguments(), [](const auto& arg) {
      return arg->template isa<ast::ReductionFunctionArgument>() ||
          arg->template isa<ast::NamedFunctionArgument>();
    }));

    for (size_t i = 0, e = call.getNumOfArguments(); i < e; ++i) {
      args.push_back(lowerBuiltInFunctionArg(
          *call.getArgument(i)->cast<ast::ExpressionFunctionArgument>()));
    }
  }

  mlir::Value CallLowerer::lowerBuiltInFunctionArg(
      const ast::FunctionArgument& arg)
  {
    auto* expressionArg = arg.cast<ast::ExpressionFunctionArgument>();
    return lowerArg(*expressionArg->getExpression());
  }

  void CallLowerer::getFunctionExpectedArgRanks(
      mlir::Operation* op,
      llvm::SmallVectorImpl<int64_t>& ranks)
  {
    assert((mlir::isa<FunctionOp, DerFunctionOp>(op)));

    if (auto functionOp = mlir::dyn_cast<FunctionOp>(op)) {
      mlir::FunctionType functionType = functionOp.getFunctionType();

      for (mlir::Type type : functionType.getInputs()) {
        if (auto arrayType = type.dyn_cast<ArrayType>()) {
          ranks.push_back(arrayType.getRank());
        } else {
          ranks.push_back(0);
        }
      }

      return;
    }

    if (auto derFunctionOp = mlir::dyn_cast<DerFunctionOp>(op)) {
      mlir::Operation* derivedFunctionOp = derFunctionOp.getOperation();

      while (derivedFunctionOp && !mlir::isa<FunctionOp>(derivedFunctionOp)) {
        derivedFunctionOp = resolveSymbolName<FunctionOp, DerFunctionOp>(
            derFunctionOp.getDerivedFunction(),
            derivedFunctionOp);
      }

      assert(derivedFunctionOp && "Derived function not found");
      auto functionOp = mlir::cast<FunctionOp>(derivedFunctionOp);

      mlir::FunctionType functionType = functionOp.getFunctionType();

      for (mlir::Type type : functionType.getInputs()) {
        if (auto arrayType = type.dyn_cast<ArrayType>()) {
          ranks.push_back(arrayType.getRank());
        } else {
          ranks.push_back(0);
        }
      }

      return;
    }
  }

  void CallLowerer::getFunctionResultTypes(
      mlir::Operation* op,
      llvm::SmallVectorImpl<mlir::Type>& types)
  {
    assert((mlir::isa<FunctionOp, DerFunctionOp>(op)));

    if (auto functionOp = mlir::dyn_cast<FunctionOp>(op)) {
      mlir::FunctionType functionType = functionOp.getFunctionType();
      auto resultTypes = functionType.getResults();
      types.append(resultTypes.begin(), resultTypes.end());
      return;
    }

    if (auto derFunctionOp = mlir::dyn_cast<DerFunctionOp>(op)) {
      mlir::Operation* derivedFunctionOp = derFunctionOp.getOperation();

      while (derivedFunctionOp && !mlir::isa<FunctionOp>(derivedFunctionOp)) {
        derivedFunctionOp = resolveSymbolName<FunctionOp, DerFunctionOp>(
            derFunctionOp.getDerivedFunction(),
            derivedFunctionOp);
      }

      assert(derivedFunctionOp && "Derived function not found");
      auto functionOp = mlir::cast<FunctionOp>(derivedFunctionOp);

      mlir::FunctionType functionType = functionOp.getFunctionType();
      auto resultTypes = functionType.getResults();
      types.append(resultTypes.begin(), resultTypes.end());

      return;
    }
  }

  bool CallLowerer::getVectorizedResultTypes(
      llvm::ArrayRef<mlir::Value> args,
      llvm::ArrayRef<int64_t> expectedArgRanks,
      llvm::ArrayRef<mlir::Type> scalarizedResultTypes,
      llvm::SmallVectorImpl<mlir::Type>& inferredResultTypes) const
  {
    assert(args.size() == expectedArgRanks.size());

    llvm::SmallVector<int64_t, 3> dimensions;

    for (size_t argIndex = 0, e = args.size(); argIndex < e; ++argIndex) {
      mlir::Value arg = args[argIndex];
      mlir::Type argType = arg.getType();
      auto argArrayType = argType.dyn_cast<ArrayType>();

      int64_t argExpectedRank = expectedArgRanks[argIndex];
      int64_t argActualRank = 0;

      if (argArrayType) {
        argActualRank = argArrayType.getRank();
      }

      if (argIndex == 0) {
        // If this is the first argument, then it will determine the
        // rank and dimensions of the result array, although the dimensions
        // can be also specialized by the other arguments if initially unknown.

        for (int64_t i = 0; i < argActualRank - argExpectedRank; ++i) {
          dimensions.push_back(argArrayType.getDimSize(i));
        }
      } else {
        // The rank difference must match with the one given by the first
        // argument, independently of the dimensions sizes.

        if (argActualRank !=
            argExpectedRank + static_cast<int64_t>(dimensions.size())) {
          return false;
        }

        for (int64_t i = 0; i < argActualRank - argExpectedRank; ++i) {
          int64_t dimension = argArrayType.getDimSize(i);

          // If the dimension is dynamic, then no further checks or
          // specializations are possible.
          if (dimension == ArrayType::kDynamic) {
            continue;
          }

          // If the dimension determined by the first argument is fixed, then
          // also the dimension of the other arguments must match (when that's
          // fixed too).

          if (dimensions[i] != ArrayType::kDynamic &&
              dimensions[i] != dimension) {
            return false;
          }

          // If the dimension determined by the first argument is dynamic, then
          // set it to a required size.
          if (dimensions[i] == ArrayType::kDynamic) {
            dimensions[i] = dimension;
          }
        }
      }
    }

    for (mlir::Type scalarizedResultType : scalarizedResultTypes) {
      llvm::SmallVector<int64_t, 3> shape;
      shape.append(dimensions);

      if (auto arrayType = scalarizedResultType.dyn_cast<ArrayType>()) {
        auto previousShape = arrayType.getShape();
        shape.append(previousShape.begin(), previousShape.end());

        inferredResultTypes.push_back(
            ArrayType::get(shape, arrayType.getElementType()));
      } else {
        if (shape.empty()) {
          inferredResultTypes.push_back(scalarizedResultType);
        } else {
          inferredResultTypes.push_back(
              ArrayType::get(shape, scalarizedResultType));
        }
      }
    }

    return true;
  }

  bool CallLowerer::isBuiltInFunction(
      const ast::ComponentReference& functionName) const
  {
    if (functionName.getPathLength() != 1) {
      return false;
    }

    if (functionName.getElement(0)->getNumOfSubscripts() != 0) {
      return false;
    }

    return llvm::StringSwitch<bool>(functionName.getElement(0)->getName())
        .Case("abs", true)
        .Case("acos", true)
        .Case("asin", true)
        .Case("atan", true)
        .Case("atan2", true)
        .Case("ceil", true)
        .Case("cos", true)
        .Case("cosh", true)
        .Case("der", true)
        .Case("diagonal", true)
        .Case("div", true)
        .Case("exp", true)
        .Case("fill", true)
        .Case("floor", true)
        .Case("identity", true)
        .Case("integer", true)
        .Case("linspace", true)
        .Case("log", true)
        .Case("log10", true)
        .Case("max", true)
        .Case("min", true)
        .Case("mod", true)
        .Case("ndims", true)
        .Case("ones", true)
        .Case("product", true)
        .Case("rem", true)
        .Case("sign", true)
        .Case("sin", true)
        .Case("sinh", true)
        .Case("size", true)
        .Case("sqrt", true)
        .Case("sum", true)
        .Case("symmetric", true)
        .Case("tan", true)
        .Case("tanh", true)
        .Case("transpose", true)
        .Case("zeros", true)
        .Default(false);
  }

  Results CallLowerer::dispatchBuiltInFunctionCall(const ast::Call& call)
  {
    auto callee = call.getCallee()->cast<ast::ComponentReference>()
        ->getElement(0)->getName();

    if (callee == "abs") {
      return abs(call);
    }

    if (callee == "acos") {
      return acos(call);
    }

    if (callee == "asin") {
      return asin(call);
    }

    if (callee == "atan") {
      return atan(call);
    }

    if (callee == "atan2") {
      return atan2(call);
    }

    if (callee == "ceil") {
      return ceil(call);
    }

    if (callee == "cos") {
      return cos(call);
    }

    if (callee == "cosh") {
      return cosh(call);
    }

    if (callee == "der") {
      return der(call);
    }

    if (callee == "diagonal") {
      return diagonal(call);
    }

    if (callee == "div") {
      return div(call);
    }

    if (callee == "exp") {
      return exp(call);
    }

    if (callee == "fill") {
      return fill(call);
    }

    if (callee == "floor") {
      return floor(call);
    }

    if (callee == "identity") {
      return identity(call);
    }

    if (callee == "integer") {
      return integer(call);
    }

    if (callee == "linspace") {
      return linspace(call);
    }

    if (callee == "log") {
      return log(call);
    }

    if (callee == "log10") {
      return log10(call);
    }

    if (callee == "max") {
      return max(call);
    }

    if (callee == "min") {
      return min(call);
    }

    if (callee == "mod") {
      return mod(call);
    }

    if (callee == "ndims") {
      return ndims(call);
    }

    if (callee == "ones") {
      return ones(call);
    }

    if (callee == "product") {
      return product(call);
    }

    if (callee == "rem") {
      return rem(call);
    }

    if (callee == "sign") {
      return sign(call);
    }

    if (callee == "sin") {
      return sin(call);
    }

    if (callee == "sinh") {
      return sinh(call);
    }

    if (callee == "size") {
      return size(call);
    }

    if (callee == "sqrt") {
      return sqrt(call);
    }

    if (callee == "sum") {
      return sum(call);
    }

    if (callee == "symmetric") {
      return symmetric(call);
    }

    if (callee == "tan") {
      return tan(call);
    }

    if (callee == "tanh") {
      return tanh(call);
    }

    if (callee == "transpose") {
      return transpose(call);
    }

    if (callee == "zeros") {
      return zeros(call);
    }

    llvm_unreachable("Unknown built-in function");
    return {};
  }

  Results CallLowerer::abs(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "abs");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(args[0].getType());

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<AbsOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::acos(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "acos");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<AcosOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::asin(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "asin");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<AsinOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::atan(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "atan");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<AtanOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::atan2(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "atan2");

    assert(call.getNumOfArguments() == 2);

    llvm::SmallVector<mlir::Value, 2> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 2> expectedArgRanks;
    expectedArgRanks.push_back(0);
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<Atan2Op>(
        loc(call.getLocation()), resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::ceil(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "ceil");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<CeilOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::cos(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "cos");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<CosOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::cosh(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "cosh");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<CoshOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::der(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "der");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<DerOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::diagonal(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "diagonal");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    auto resultType = ArrayType::get(
        {ArrayType::kDynamic, ArrayType::kDynamic},
        IntegerType::get(builder().getContext()));

    mlir::Value result = builder().create<DiagonalOp>(
        loc(call.getLocation()), resultType, args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::div(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "div");

    assert(call.getNumOfArguments() == 2);

    llvm::SmallVector<mlir::Value, 2> args;
    lowerBuiltInFunctionArgs(call, args);

    mlir::Type resultType = IntegerType::get(builder().getContext());

    if (args[0].getType().isa<RealType>() ||
        args[1].getType().isa<RealType>()) {
      resultType = RealType::get(builder().getContext());
    }

    mlir::Value result = builder().create<DivTruncOp>(
        loc(call.getLocation()), resultType, args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::exp(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "exp");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<ExpOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::fill(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "fill");

    assert(call.getNumOfArguments() > 0);

    assert(call.getArgument(0)->isa<ast::ExpressionFunctionArgument>());

    mlir::Value value = lowerArg(
        *call.getArgument(0)
             ->cast<ast::ExpressionFunctionArgument>()
             ->getExpression());

    llvm::SmallVector<int64_t, 1> shape;

    for (size_t i = 1, e = call.getNumOfArguments(); i < e; ++i) {
      assert(call.getArgument(i)->isa<ast::ExpressionFunctionArgument>());

      const ast::Expression* arg =
          call.getArgument(i)
              ->cast<ast::ExpressionFunctionArgument>()
              ->getExpression();

      assert(arg->isa<ast::Constant>());
      shape.push_back(arg->cast<ast::Constant>()->as<int64_t>());
    }

    auto resultType = ArrayType::get(
        shape, IntegerType::get(builder().getContext()));

    mlir::Value result = builder().create<FillOp>(
        loc(call.getLocation()), resultType, value);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::floor(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "floor");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<FloorOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::identity(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "identity");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    auto resultType = ArrayType::get(
        {ArrayType::kDynamic, ArrayType::kDynamic},
        IntegerType::get(builder().getContext()));

    mlir::Value result = builder().create<IdentityOp>(
        loc(call.getLocation()), resultType, args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::integer(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "integer");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(IntegerType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<IntegerOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::linspace(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "linspace");

    assert(call.getNumOfArguments() == 3);

    llvm::SmallVector<mlir::Value, 3> args;
    lowerBuiltInFunctionArgs(call, args);

    auto resultType = ArrayType::get(
        ArrayType::kDynamic, RealType::get(builder().getContext()));

    mlir::Value result = builder().create<LinspaceOp>(
        loc(call.getLocation()), resultType, args[0], args[1], args[2]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::log(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "log");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<LogOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::log10(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "log10");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<Log10Op>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::max(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "max");

    size_t numOfArguments = call.getNumOfArguments();
    assert(numOfArguments == 1 || numOfArguments == 2);

    if (numOfArguments == 1) {
      if (call.getArgument(0)->isa<ast::ReductionFunctionArgument>()) {
        return maxReduction(call);
      }

      return maxArray(call);
    }

    return maxScalars(call);
  }

  Results CallLowerer::maxArray(const ast::Call& call)
  {
    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    mlir::Type resultType =
        args[0].getType().cast<ArrayType>().getElementType();

    mlir::Value result = builder().create<MaxOp>(
        loc(call.getLocation()), resultType, args);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::maxReduction(const ast::Call& call)
  {
    return reduction(call, "max");
  }

  Results CallLowerer::maxScalars(const ast::Call& call)
  {
    assert(call.getNumOfArguments() == 2);

    llvm::SmallVector<mlir::Value, 2> args;
    lowerBuiltInFunctionArgs(call, args);

    mlir::Type resultType = getMostGenericScalarType(
        args[0].getType(), args[1].getType());

    mlir::Value result = builder().create<MaxOp>(
        loc(call.getLocation()), resultType, args);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::min(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "min");

    size_t numOfArguments = call.getNumOfArguments();
    assert(numOfArguments == 1 || numOfArguments == 2);

    if (numOfArguments == 1) {
      if (call.getArgument(0)->isa<ast::ReductionFunctionArgument>()) {
        return minReduction(call);
      }

      return minArray(call);
    }

    return minScalars(call);
  }

  Results CallLowerer::minArray(const ast::Call& call)
  {
    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    mlir::Type resultType =
        args[0].getType().cast<ArrayType>().getElementType();

    mlir::Value result = builder().create<MinOp>(
        loc(call.getLocation()), resultType, args);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::minReduction(const ast::Call& call)
  {
    return reduction(call, "min");
  }

  Results CallLowerer::minScalars(const ast::Call& call)
  {
    assert(call.getNumOfArguments() == 2);

    llvm::SmallVector<mlir::Value, 2> args;
    lowerBuiltInFunctionArgs(call, args);

    mlir::Type resultType = getMostGenericScalarType(
        args[0].getType(), args[1].getType());

    mlir::Value result = builder().create<MinOp>(
        loc(call.getLocation()), resultType, args);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::mod(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "mod");

    assert(call.getNumOfArguments() == 2);

    llvm::SmallVector<mlir::Value, 2> args;
    lowerBuiltInFunctionArgs(call, args);

    mlir::Type resultType = IntegerType::get(builder().getContext());

    if (args[0].getType().isa<RealType>() ||
        args[1].getType().isa<RealType>()) {
      resultType = RealType::get(builder().getContext());
    }

    mlir::Value result = builder().create<ModOp>(
        loc(call.getLocation()), resultType, args);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::ndims(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "ndims");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    auto resultType = IntegerType::get(builder().getContext());

    mlir::Value result = builder().create<NDimsOp>(
        loc(call.getLocation()), resultType, args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::ones(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "ones");

    assert(call.getNumOfArguments() > 0);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> shape(args.size(), ArrayType::kDynamic);

    auto resultType = ArrayType::get(
        shape, IntegerType::get(builder().getContext()));

    mlir::Value result = builder().create<OnesOp>(
        loc(call.getLocation()), resultType, args);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::product(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "product");

    assert(call.getNumOfArguments() == 1);

    if (call.getArgument(0)->isa<ast::ReductionFunctionArgument>()) {
      return productReduction(call);
    }

    return productArray(call);
  }

  Results CallLowerer::productArray(const ast::Call& call)
  {
    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    auto argArrayType = args[0].getType().cast<ArrayType>();
    mlir::Type resultType = argArrayType.getElementType();

    mlir::Value result = builder().create<ProductOp>(
        loc(call.getLocation()), resultType, args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::productReduction(const ast::Call& call)
  {
    return reduction(call, "mul");
  }

  Results CallLowerer::rem(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "rem");

    assert(call.getNumOfArguments() == 2);

    llvm::SmallVector<mlir::Value, 2> args;
    lowerBuiltInFunctionArgs(call, args);

    mlir::Type resultType = IntegerType::get(builder().getContext());

    if (args[0].getType().isa<RealType>() ||
        args[1].getType().isa<RealType>()) {
      resultType = RealType::get(builder().getContext());
    }

    mlir::Value result = builder().create<RemOp>(
        loc(call.getLocation()), resultType, args);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::sign(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "sign");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    auto resultType = IntegerType::get(builder().getContext());

    mlir::Value result = builder().create<SignOp>(
        loc(call.getLocation()), resultType, args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::sin(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() == "sin");
    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<SinOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::sinh(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "sinh");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<SinhOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::size(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "size");

    assert(call.getNumOfArguments() == 1 || call.getNumOfArguments() == 2);

    llvm::SmallVector<mlir::Value, 2> args;
    lowerBuiltInFunctionArgs(call, args);

    if (args.size() == 1) {
      mlir::Type resultType = ArrayType::get(
          args[0].getType().cast<ArrayType>().getRank(),
          IntegerType::get(builder().getContext()));

      mlir::Value result = builder().create<SizeOp>(
          loc(call.getLocation()), resultType, args);

      return Reference::ssa(builder(), result);
    }

    mlir::Type resultType = IntegerType::get(builder().getContext());

    mlir::Value oneValue = builder().create<ConstantOp>(
        args[1].getLoc(), IntegerAttr::get(builder().getContext(), 1));

    mlir::Value index = builder().create<SubOp>(
        args[1].getLoc(), builder().getIndexType(), args[1], oneValue);

    mlir::Value result = builder().create<SizeOp>(
        loc(call.getLocation()), resultType, args[0], index);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::sqrt(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "sqrt");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<SqrtOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::sum(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "sum");

    assert(call.getNumOfArguments() == 1);

    if (call.getArgument(0)->isa<ast::ReductionFunctionArgument>()) {
      return sumReduction(call);
    }

    return sumArray(call);
  }

  Results CallLowerer::sumArray(const ast::Call& call)
  {
    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    auto argArrayType = args[0].getType().cast<ArrayType>();
    mlir::Type resultType = argArrayType.getElementType();

    mlir::Value result = builder().create<SumOp>(
        loc(call.getLocation()), resultType, args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::sumReduction(const ast::Call& call)
  {
    return reduction(call, "add");
  }

  Results CallLowerer::symmetric(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "symmetric");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    mlir::Type resultType = args[0].getType();

    mlir::Value result = builder().create<SymmetricOp>(
        loc(call.getLocation()), resultType, args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::tan(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "tan");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<TanOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::tanh(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "tanh");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> expectedArgRanks;
    expectedArgRanks.push_back(0);

    llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
    scalarizedResultTypes.push_back(RealType::get(builder().getContext()));

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (!getVectorizedResultTypes(
            args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
      assert("Can't vectorize function call");
      return {};
    }

    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<TanhOp>(
        loc(call.getLocation()), resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::transpose(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "transpose");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 2> shape;
    auto argArrayType = args[0].getType().cast<ArrayType>();
    shape.push_back(argArrayType.getDimSize(1));
    shape.push_back(argArrayType.getDimSize(0));
    auto resultType = ArrayType::get(shape, argArrayType.getElementType());

    mlir::Value result = builder().create<TransposeOp>(
        loc(call.getLocation()), resultType, args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::zeros(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ComponentReference>()->getName() ==
           "zeros");

    assert(call.getNumOfArguments() > 0);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerBuiltInFunctionArgs(call, args);

    llvm::SmallVector<int64_t, 1> shape(args.size(), ArrayType::kDynamic);

    auto resultType = ArrayType::get(
        shape, IntegerType::get(builder().getContext()));

    mlir::Value result = builder().create<ZerosOp>(
        loc(call.getLocation()), resultType, args);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::reduction(const ast::Call& call, llvm::StringRef action)
  {
    assert(call.getNumOfArguments() == 1);
    mlir::Type resultType = RealType::get(builder().getContext());

    auto* reductionArg =
        call.getArgument(0)->cast<ast::ReductionFunctionArgument>();

    // Lower the iteration ranges.
    llvm::SmallVector<mlir::Value, 3> iterables;

    for (size_t i = 0, e = reductionArg->getNumOfForIndices(); i < e; ++i) {
      Results inductionResults =
          lower(*reductionArg->getForIndex(i)->getExpression());

      assert(inductionResults.size() == 1);

      iterables.push_back(
          inductionResults[0].get(inductionResults[0].getLoc()));
    }

    // Create the operation.
    auto reductionOp = builder().create<ReductionOp>(
        loc(call.getLocation()), resultType, action, iterables);

    mlir::OpBuilder::InsertionGuard guard(builder());

    builder().setInsertionPointToStart(
        reductionOp.createExpressionBlock(builder()));

    // Map the induction variables.
    assert(reductionArg->getNumOfForIndices() ==
           reductionOp.getInductions().size());

    Lowerer::VariablesScope scope(getVariablesSymbolTable());

    for (size_t i = 0, e = reductionArg->getNumOfForIndices(); i < e; ++i) {
      getVariablesSymbolTable().insert(
          reductionArg->getForIndex(i)->getName(),
          Reference::ssa(builder(), reductionOp.getInductions()[i]));
    }

    // Lower the expression.
    Results expressionResults = lower(*reductionArg->getExpression());
    assert(expressionResults.size() == 1);

    mlir::Value result =
        expressionResults[0].get(expressionResults[0].getLoc());

    if (result.getType() != resultType) {
      result = builder().create<CastOp>(result.getLoc(), resultType, result);
    }

    builder().create<YieldOp>(result.getLoc(), result);
    return Reference::ssa(builder(), reductionOp.getResult());
  }
}
