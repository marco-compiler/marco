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
    const ast::Expression* function = call.getCallee();

    llvm::StringRef functionName =
        function->dyn_cast<ast::ReferenceAccess>()->getName();

    auto caller = mlir::cast<ClassInterface>(
        getClass(*call.getParentOfType<ast::Class>()));

    mlir::Operation* calleeOp =
        resolveSymbolName<FunctionOp, DerFunctionOp>(functionName, caller);

    if (calleeOp != nullptr) {
      // User-defined function.
      llvm::SmallVector<mlir::Value, 3> args;
      lowerArgs(call, args);

      llvm::SmallVector<int64_t, 3> expectedArgRanks;
      getFunctionExpectedArgRanks(calleeOp, expectedArgRanks);

      llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
      getFunctionResultTypes(calleeOp, scalarizedResultTypes);

      llvm::SmallVector<mlir::Type, 1> resultTypes;

      if (!getVectorizedResultTypes(
              args, expectedArgRanks, scalarizedResultTypes, resultTypes)) {
        assert("Can't vectorize function call");
        return {};
      }

      auto callOp = builder().create<CallOp>(
          loc(call.getLocation()), functionName, resultTypes, args);

      std::vector<Reference> results;

      for (auto result : callOp->getResults()) {
        results.push_back(Reference::ssa(builder(), result));
      }

      return Results(results.begin(), results.end());
    }

    if (isBuiltInFunction(functionName)) {
      // Built-in function.
      return dispatchBuiltInFunctionCall(call);
    }

    // The function doesn't exist.
    llvm_unreachable("Function not found");
    return {};
  }

  mlir::Value CallLowerer::lowerArg(const ast::Expression& expression)
  {
    mlir::Location location = loc(expression.getLocation());
    auto results = lower(expression);
    assert(results.size() == 1);
    return results[0].get(location);
  }

  void CallLowerer::lowerArgs(
      const ast::Call& call,
      llvm::SmallVectorImpl<mlir::Value>& args)
  {
    for (size_t i = 0, e = call.getNumOfArguments(); i < e; ++i) {
      args.push_back(lowerArg(*call.getArgument(i)));
    }
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
          if (dimension == ArrayType::kDynamicSize) {
            continue;
          }

          // If the dimension determined by the first argument is fixed, then
          // also the dimension of the other arguments must match (when that's
          // fixed too).

          if (dimensions[i] != ArrayType::kDynamicSize &&
              dimensions[i] != dimension) {
            return false;
          }

          // If the dimension determined by the first argument is dynamic, then
          // set it to a required size.
          if (dimensions[i] == ArrayType::kDynamicSize) {
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

  bool CallLowerer::isBuiltInFunction(llvm::StringRef name) const
  {
    return llvm::StringSwitch<bool>(name)
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
    llvm::StringRef functionName =
        call.getCallee()->cast<ast::ReferenceAccess>()->getName();

    if (functionName == "abs") {
      return abs(call);
    }

    if (functionName == "acos") {
      return acos(call);
    }

    if (functionName == "asin") {
      return asin(call);
    }

    if (functionName == "atan") {
      return atan(call);
    }

    if (functionName == "atan2") {
      return atan2(call);
    }

    if (functionName == "ceil") {
      return ceil(call);
    }

    if (functionName == "cos") {
      return cos(call);
    }

    if (functionName == "cosh") {
      return cosh(call);
    }

    if (functionName == "der") {
      return der(call);
    }

    if (functionName == "diagonal") {
      return diagonal(call);
    }

    if (functionName == "div") {
      return div(call);
    }

    if (functionName == "exp") {
      return exp(call);
    }

    if (functionName == "floor") {
      return floor(call);
    }

    if (functionName == "identity") {
      return identity(call);
    }

    if (functionName == "integer") {
      return integer(call);
    }

    if (functionName == "linspace") {
      return linspace(call);
    }

    if (functionName == "log") {
      return log(call);
    }

    if (functionName == "log10") {
      return log10(call);
    }

    if (functionName == "max") {
      return max(call);
    }

    if (functionName == "min") {
      return min(call);
    }

    if (functionName == "mod") {
      return mod(call);
    }

    if (functionName == "ndims") {
      return ndims(call);
    }

    if (functionName == "ones") {
      return ones(call);
    }

    if (functionName == "product") {
      return product(call);
    }

    if (functionName == "rem") {
      return rem(call);
    }

    if (functionName == "sign") {
      return sign(call);
    }

    if (functionName == "sin") {
      return sin(call);
    }

    if (functionName == "sinh") {
      return sinh(call);
    }

    if (functionName == "size") {
      return size(call);
    }

    if (functionName == "sqrt") {
      return sqrt(call);
    }

    if (functionName == "sum") {
      return sum(call);
    }

    if (functionName == "symmetric") {
      return symmetric(call);
    }

    if (functionName == "tan") {
      return tan(call);
    }

    if (functionName == "tanh") {
      return tanh(call);
    }

    if (functionName == "transpose") {
      return transpose(call);
    }

    if (functionName == "zeros") {
      return zeros(call);
    }

    llvm_unreachable("Unknown built-in function");
    return {};
  }

  Results CallLowerer::abs(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() == "abs");
    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "acos");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "asin");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "atan");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "atan2");

    assert(call.getNumOfArguments() == 2);

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "ceil");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "cos");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "cosh");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() == "der");
    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "diagonal");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

    auto resultType = ArrayType::get(
        {-1, -1}, IntegerType::get(builder().getContext()));

    mlir::Value result = builder().create<DiagonalOp>(
        loc(call.getLocation()), resultType, args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::div(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() == "div");
    assert(call.getNumOfArguments() == 2);

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() == "exp");
    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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

  Results CallLowerer::floor(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "floor");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() == "identity");
    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

    auto resultType = ArrayType::get(
        {-1, -1}, IntegerType::get(builder().getContext()));

    mlir::Value result = builder().create<IdentityOp>(
        loc(call.getLocation()), resultType, args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::integer(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "integer");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "linspace");

    assert(call.getNumOfArguments() == 3);

    llvm::SmallVector<mlir::Value, 3> args;
    lowerArgs(call, args);

    auto resultType = ArrayType::get(
        ArrayType::kDynamicSize, RealType::get(builder().getContext()));

    mlir::Value result = builder().create<LinspaceOp>(
        loc(call.getLocation()), resultType, args[0], args[1], args[2]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::log(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() == "log");
    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "log10");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() == "max");
    assert(call.getNumOfArguments() == 1 || call.getNumOfArguments() == 2);

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(call, args);

    mlir::Type resultType;

    if (args.size() == 1) {
      resultType = args[0].getType().cast<ArrayType>().getElementType();
    } else {
      resultType = getMostGenericScalarType(
          args[0].getType(), args[1].getType());
    }

    mlir::Value result = builder().create<MaxOp>(
        loc(call.getLocation()), resultType, args);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::min(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() == "min");
    assert(call.getNumOfArguments() == 1 || call.getNumOfArguments() == 2);

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(call, args);

    mlir::Type resultType;

    if (args.size() == 1) {
      resultType = args[0].getType().cast<ArrayType>().getElementType();
    } else {
      resultType = getMostGenericScalarType(
          args[0].getType(), args[1].getType());
    }

    mlir::Value result = builder().create<MinOp>(
        loc(call.getLocation()), resultType, args);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::mod(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() == "mod");
    assert(call.getNumOfArguments() == 2);

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "ndims");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

    auto resultType = IntegerType::get(builder().getContext());

    mlir::Value result = builder().create<NDimsOp>(
        loc(call.getLocation()), resultType, args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::ones(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "ones");

    assert(call.getNumOfArguments() > 0);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

    llvm::SmallVector<int64_t, 1> shape(args.size(), ArrayType::kDynamicSize);

    auto resultType = ArrayType::get(
        shape, IntegerType::get(builder().getContext()));

    mlir::Value result = builder().create<OnesOp>(
        loc(call.getLocation()), resultType, args);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::product(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "product");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

    auto argArrayType = args[0].getType().cast<ArrayType>();
    mlir::Type resultType = argArrayType.getElementType();

    mlir::Value result = builder().create<ProductOp>(
        loc(call.getLocation()), resultType, args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::rem(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() == "rem");
    assert(call.getNumOfArguments() == 2);

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "sign");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

    auto resultType = IntegerType::get(builder().getContext());

    mlir::Value result = builder().create<SignOp>(
        loc(call.getLocation()), resultType, args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::sin(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() == "sin");
    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "sinh");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "size");

    assert(call.getNumOfArguments() == 1 || call.getNumOfArguments() == 2);

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "sqrt");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() == "sum");
    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

    auto argArrayType = args[0].getType().cast<ArrayType>();
    mlir::Type resultType = argArrayType.getElementType();

    mlir::Value result = builder().create<SumOp>(
        loc(call.getLocation()), resultType, args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::symmetric(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "symmetric");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

    mlir::Type resultType = args[0].getType();

    mlir::Value result = builder().create<SymmetricOp>(
        loc(call.getLocation()), resultType, args[0]);

    return Reference::ssa(builder(), result);
  }

  Results CallLowerer::tan(const ast::Call& call)
  {
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() == "tan");
    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "tanh");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "transpose");

    assert(call.getNumOfArguments() == 1);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

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
    assert(call.getCallee()->cast<ast::ReferenceAccess>()->getName() ==
           "zeros");

    assert(call.getNumOfArguments() > 0);

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(call, args);

    llvm::SmallVector<int64_t, 1> shape(args.size(), ArrayType::kDynamicSize);

    auto resultType = ArrayType::get(
        shape, IntegerType::get(builder().getContext()));

    mlir::Value result = builder().create<ZerosOp>(
        loc(call.getLocation()), resultType, args);

    return Reference::ssa(builder(), result);
  }
}
