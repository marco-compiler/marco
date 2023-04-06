#include "marco/Codegen/Lowering/OperationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen::lowering;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  OperationLowerer::OperationLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  Results OperationLowerer::lower(const ast::Operation& operation)
  {
    auto lowererFn = [](ast::OperationKind kind) -> OperationLowerer::LoweringFunction {
      switch (kind) {
        case ast::OperationKind::negate:
          return &OperationLowerer::negate;

        case ast::OperationKind::add:
          return &OperationLowerer::add;

        case ast::OperationKind::addEW:
          return &OperationLowerer::addEW;

        case ast::OperationKind::subtract:
          return &OperationLowerer::subtract;

        case ast::OperationKind::subtractEW:
          return &OperationLowerer::subtractEW;

        case ast::OperationKind::multiply:
          return &OperationLowerer::multiply;

        case ast::OperationKind::multiplyEW:
          return &OperationLowerer::multiplyEW;

        case ast::OperationKind::divide:
          return &OperationLowerer::divide;

        case ast::OperationKind::divideEW:
          return &OperationLowerer::divideEW;

        case ast::OperationKind::ifelse:
          return &OperationLowerer::ifElse;

        case ast::OperationKind::greater:
          return &OperationLowerer::greater;

        case ast::OperationKind::greaterEqual:
          return &OperationLowerer::greaterOrEqual;

        case ast::OperationKind::equal:
          return &OperationLowerer::equal;

        case ast::OperationKind::different:
          return &OperationLowerer::notEqual;

        case ast::OperationKind::lessEqual:
          return &OperationLowerer::lessOrEqual;

        case ast::OperationKind::less:
          return &OperationLowerer::less;

        case ast::OperationKind::land:
          return &OperationLowerer::logicalAnd;

        case ast::OperationKind::lnot:
          return &OperationLowerer::logicalNot;

        case ast::OperationKind::lor:
          return &OperationLowerer::logicalOr;

        case ast::OperationKind::subscription:
          return &OperationLowerer::subscription;

        case ast::OperationKind::memberLookup:
          return &OperationLowerer::memberLookup;

        case ast::OperationKind::powerOf:
          return &OperationLowerer::powerOf;

        case ast::OperationKind::powerOfEW:
          return &OperationLowerer::powerOfEW;

        default:
          return nullptr;
      }
    };

    auto lowerer = lowererFn(operation.getOperationKind());
    assert(lowerer != nullptr && "Unknown operation type");
    return lowerer(*this, operation);
  }

  mlir::Value OperationLowerer::lowerArg(const ast::Expression& expression)
  {
    mlir::Location location = loc(expression.getLocation());
    auto results = lower(expression);
    assert(results.size() == 1);
    return results[0].get(location);
  }

  void OperationLowerer::lowerArgs(
      const ast::Operation& operation,
      llvm::SmallVectorImpl<mlir::Value>& args)
  {
    for (size_t i = 0, e = operation.getNumOfArguments(); i < e; ++i) {
      args.push_back(lowerArg(*operation.getArgument(i)));
    }
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::negate>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    assert(operands.size() == 1);
    mlir::Type operandType = operands[0].getType();

    if (isScalarType(operandType)) {
      inferredTypes.push_back(operandType);
      return true;
    }

    if (auto arrayType = operandType.dyn_cast<ArrayType>()) {
      inferredTypes.push_back(arrayType);
      return true;
    }

    return false;
  }

  Results OperationLowerer::negate(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 1> args;
    lowerArgs(operation, args);
    assert(args.size() == 1);

    mlir::Value result = builder().create<NegateOp>(
        location, args[0].getType(), args[0]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::add>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    assert(operands.size() == 2);
    mlir::Type lhsType = operands[0].getType();
    mlir::Type rhsType = operands[1].getType();

    if (isScalarType(lhsType) && isScalarType(rhsType)) {
      inferredTypes.push_back(getMostGenericScalarType(lhsType, rhsType));
      return true;
    }

    auto lhsArrayType = lhsType.dyn_cast<ArrayType>();
    auto rhsArrayType = rhsType.dyn_cast<ArrayType>();

    if (lhsArrayType && rhsArrayType) {
      if (lhsArrayType.getRank() != rhsArrayType.getRank()) {
        return false;
      }

      for (const auto& [lhsDim, rhsDim] :
           llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        if (lhsDim != ArrayType::kDynamicSize &&
            rhsDim != ArrayType::kDynamicSize &&
            lhsDim != rhsDim) {
          return false;
        }
      }

      mlir::Type resultElementType = getMostGenericScalarType(
          lhsArrayType.getElementType(), rhsArrayType.getElementType());

      inferredTypes.push_back(
          ArrayType::get(lhsArrayType.getShape(), resultElementType));

      return true;
    }

    return false;
  }

  Results OperationLowerer::add(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() >= 2);

    llvm::SmallVector<mlir::Value, 2> current;
    current.push_back(args[0]);
    current.push_back(args[1]);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::add>(
        builder().getContext(), current, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<AddOp>(
        location, resultTypes[0], current[0], current[1]);

    for (size_t i = 2; i < args.size(); ++i) {
      current.clear();
      args.push_back(result);
      args.push_back(args[i]);

      resultTypes.clear();

      inferResult = inferResultTypes<ast::OperationKind::add>(
          builder().getContext(), current, resultTypes);

      assert(inferResult && "Can't infer result type");
      assert(resultTypes.size() == 1);

      result = builder().create<AddOp>(
          location, resultTypes[0], current[0], current[1]);
    }

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::addEW>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    assert(operands.size() == 2);
    mlir::Type lhsType = operands[0].getType();
    mlir::Type rhsType = operands[1].getType();

    if (isScalarType(lhsType) && isScalarType(rhsType)) {
      inferredTypes.push_back(getMostGenericScalarType(lhsType, rhsType));
      return true;
    }

    auto lhsArrayType = lhsType.dyn_cast<ArrayType>();
    auto rhsArrayType = rhsType.dyn_cast<ArrayType>();

    if (isScalarType(lhsType) && rhsArrayType) {
      mlir::Type resultElementType = getMostGenericScalarType(
          lhsType, rhsArrayType.getElementType());

      inferredTypes.push_back(
          rhsArrayType.toElementType(resultElementType));

      return true;
    }

    if (lhsArrayType && isScalarType(rhsType)) {
      mlir::Type resultElementType = getMostGenericScalarType(
          lhsArrayType.getElementType(), rhsType);

      inferredTypes.push_back(
          lhsArrayType.toElementType(resultElementType));

      return true;
    }

    if (lhsArrayType && rhsArrayType) {
      if (lhsArrayType.getRank() != rhsArrayType.getRank()) {
        return false;
      }

      llvm::SmallVector<int64_t> shape;

      for (const auto& [lhsDim, rhsDim] :
           llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        if (lhsDim != ArrayType::kDynamicSize) {
          shape.push_back(lhsDim);
        } else if (rhsDim != ArrayType::kDynamicSize) {
          shape.push_back(rhsDim);
        } else {
          shape.push_back(ArrayType::kDynamicSize);
        }
      }

      mlir::Type resultElementType = getMostGenericScalarType(
          lhsArrayType.getElementType(), rhsArrayType.getElementType());

      inferredTypes.push_back(ArrayType::get(shape, resultElementType));
      return true;
    }

    return false;
  }

  Results OperationLowerer::addEW(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::addEW>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<AddEWOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::subtract>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    assert(operands.size() == 2);
    mlir::Type lhsType = operands[0].getType();
    mlir::Type rhsType = operands[1].getType();

    if (isScalarType(lhsType) && isScalarType(rhsType)) {
      inferredTypes.push_back(getMostGenericScalarType(lhsType, rhsType));
      return true;
    }

    auto lhsArrayType = lhsType.dyn_cast<ArrayType>();
    auto rhsArrayType = rhsType.dyn_cast<ArrayType>();

    if (lhsArrayType && rhsArrayType) {
      if (lhsArrayType.getRank() != rhsArrayType.getRank()) {
        return false;
      }

      for (const auto& [lhsDim, rhsDim] :
           llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        if (lhsDim != ArrayType::kDynamicSize &&
            rhsDim != ArrayType::kDynamicSize &&
            lhsDim != rhsDim) {
          return false;
        }
      }

      mlir::Type resultElementType = getMostGenericScalarType(
          lhsArrayType.getElementType(), rhsArrayType.getElementType());

      inferredTypes.push_back(
          ArrayType::get(lhsArrayType.getShape(), resultElementType));

      return true;
    }

    return false;
  }

  Results OperationLowerer::subtract(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::subtract>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<SubOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::subtractEW>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    assert(operands.size() == 2);
    mlir::Type lhsType = operands[0].getType();
    mlir::Type rhsType = operands[1].getType();

    if (isScalarType(lhsType) && isScalarType(rhsType)) {
      inferredTypes.push_back(getMostGenericScalarType(lhsType, rhsType));
      return true;
    }

    auto lhsArrayType = lhsType.dyn_cast<ArrayType>();
    auto rhsArrayType = rhsType.dyn_cast<ArrayType>();

    if (isScalarType(lhsType) && rhsArrayType) {
      mlir::Type resultElementType = getMostGenericScalarType(
          lhsType, rhsArrayType.getElementType());

      inferredTypes.push_back(
          rhsArrayType.toElementType(resultElementType));

      return true;
    }

    if (lhsArrayType && isScalarType(rhsType)) {
      mlir::Type resultElementType = getMostGenericScalarType(
          lhsArrayType.getElementType(), rhsType);

      inferredTypes.push_back(
          lhsArrayType.toElementType(resultElementType));

      return true;
    }

    if (lhsArrayType && rhsArrayType) {
      if (lhsArrayType.getRank() != rhsArrayType.getRank()) {
        return false;
      }

      llvm::SmallVector<int64_t> shape;

      for (const auto& [lhsDim, rhsDim] :
           llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        if (lhsDim != ArrayType::kDynamicSize) {
          shape.push_back(lhsDim);
        } else if (rhsDim != ArrayType::kDynamicSize) {
          shape.push_back(rhsDim);
        } else {
          shape.push_back(ArrayType::kDynamicSize);
        }
      }

      mlir::Type resultElementType = getMostGenericScalarType(
          lhsArrayType.getElementType(), rhsArrayType.getElementType());

      inferredTypes.push_back(ArrayType::get(shape, resultElementType));
      return true;
    }

    return false;
  }

  Results OperationLowerer::subtractEW(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::subtractEW>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<SubEWOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::multiply>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    assert(operands.size() == 2);
    mlir::Type lhsType = operands[0].getType();
    mlir::Type rhsType = operands[1].getType();

    if (isScalarType(lhsType) && isScalarType(rhsType)) {
      inferredTypes.push_back(getMostGenericScalarType(lhsType, rhsType));
      return true;
    }

    auto lhsArrayType = lhsType.dyn_cast<ArrayType>();
    auto rhsArrayType = rhsType.dyn_cast<ArrayType>();

    if (isScalarType(lhsType) && rhsArrayType) {
      mlir::Type resultElementType = getMostGenericScalarType(
          lhsType, rhsArrayType.getElementType());

      inferredTypes.push_back(
          rhsArrayType.toElementType(resultElementType));

      return true;
    }

    if (lhsArrayType && rhsArrayType) {
      mlir::Type resultElementType = getMostGenericScalarType(
          lhsArrayType.getElementType(), rhsArrayType.getElementType());

      if (lhsArrayType.getRank() == 1 && rhsArrayType.getRank() == 1) {
        inferredTypes.push_back(resultElementType);
        return true;
      }

      if (lhsArrayType.getRank() == 1 && rhsArrayType.getRank() == 2) {
        inferredTypes.push_back(
            ArrayType::get(rhsArrayType.getShape()[1], resultElementType));

        return true;
      }

      if (lhsArrayType.getRank() == 2 && rhsArrayType.getRank() == 1) {
        inferredTypes.push_back(
            ArrayType::get(lhsArrayType.getShape()[0], resultElementType));

        return true;
      }

      if (lhsArrayType.getRank() == 2 && rhsArrayType.getRank() == 2) {
        llvm::SmallVector<int64_t> shape;
        shape.push_back(lhsArrayType.getShape()[0]);
        shape.push_back(rhsArrayType.getShape()[1]);

        inferredTypes.push_back(
            ArrayType::get(shape, resultElementType));

        return true;
      }

      return false;
    }

    return false;
  }

  Results OperationLowerer::multiply(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() >= 2);

    llvm::SmallVector<mlir::Value, 2> current;
    current.push_back(args[0]);
    current.push_back(args[1]);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::multiply>(
        builder().getContext(), current, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<MulOp>(
        location, resultTypes[0], current[0], current[1]);

    for (size_t i = 2; i < args.size(); ++i) {
      current.clear();
      args.push_back(result);
      args.push_back(args[i]);

      resultTypes.clear();

      inferResult = inferResultTypes<ast::OperationKind::multiply>(
          builder().getContext(), current, resultTypes);

      assert(inferResult && "Can't infer result type");
      assert(resultTypes.size() == 1);

      result = builder().create<MulOp>(
          location, resultTypes[0], current[0], current[1]);
    }

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::multiplyEW>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    assert(operands.size() == 2);
    mlir::Type lhsType = operands[0].getType();
    mlir::Type rhsType = operands[1].getType();

    if (isScalarType(lhsType) && isScalarType(rhsType)) {
      inferredTypes.push_back(getMostGenericScalarType(lhsType, rhsType));
      return true;
    }

    auto lhsArrayType = lhsType.dyn_cast<ArrayType>();
    auto rhsArrayType = rhsType.dyn_cast<ArrayType>();

    if (isScalarType(lhsType) && rhsArrayType) {
      mlir::Type resultElementType = getMostGenericScalarType(
          lhsType, rhsArrayType.getElementType());

      inferredTypes.push_back(
          rhsArrayType.toElementType(resultElementType));

      return true;
    }

    if (lhsArrayType && isScalarType(rhsType)) {
      mlir::Type resultElementType = getMostGenericScalarType(
          lhsArrayType.getElementType(), rhsType);

      inferredTypes.push_back(
          lhsArrayType.toElementType(resultElementType));

      return true;
    }

    if (lhsArrayType && rhsArrayType) {
      if (lhsArrayType.getRank() != rhsArrayType.getRank()) {
        return false;
      }

      llvm::SmallVector<int64_t> shape;

      for (const auto& [lhsDim, rhsDim] :
           llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        if (lhsDim != ArrayType::kDynamicSize) {
          shape.push_back(lhsDim);
        } else if (rhsDim != ArrayType::kDynamicSize) {
          shape.push_back(rhsDim);
        } else {
          shape.push_back(ArrayType::kDynamicSize);
        }
      }

      mlir::Type resultElementType = getMostGenericScalarType(
          lhsArrayType.getElementType(), rhsArrayType.getElementType());

      inferredTypes.push_back(ArrayType::get(shape, resultElementType));
      return true;
    }

    return false;
  }

  Results OperationLowerer::multiplyEW(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::multiplyEW>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<MulEWOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::divide>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    assert(operands.size() == 2);
    mlir::Type lhsType = operands[0].getType();
    mlir::Type rhsType = operands[1].getType();

    if (isScalarType(lhsType) && isScalarType(rhsType)) {
      inferredTypes.push_back(
          getMostGenericScalarType(lhsType, rhsType));

      return true;
    }

    auto lhsArrayType = lhsType.dyn_cast<ArrayType>();

    if (lhsArrayType && isScalarType(rhsType)) {
      mlir::Type resultElementType = getMostGenericScalarType(
          lhsArrayType.getElementType(), rhsType);

      inferredTypes.push_back(
          lhsArrayType.toElementType(resultElementType));

      return true;
    }

    return false;
  }

  Results OperationLowerer::divide(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::divide>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<DivOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::divideEW>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    assert(operands.size() == 2);
    mlir::Type lhsType = operands[0].getType();
    mlir::Type rhsType = operands[1].getType();

    if (isScalarType(lhsType) && isScalarType(rhsType)) {
      inferredTypes.push_back(getMostGenericScalarType(lhsType, rhsType));
      return true;
    }

    auto lhsArrayType = lhsType.dyn_cast<ArrayType>();
    auto rhsArrayType = rhsType.dyn_cast<ArrayType>();

    if (isScalarType(lhsType) && rhsArrayType) {
      mlir::Type resultElementType = getMostGenericScalarType(
          lhsType, rhsArrayType.getElementType());

      inferredTypes.push_back(
          rhsArrayType.toElementType(resultElementType));

      return true;
    }

    if (lhsArrayType && isScalarType(rhsType)) {
      mlir::Type resultElementType = getMostGenericScalarType(
          lhsArrayType.getElementType(), rhsType);

      inferredTypes.push_back(
          lhsArrayType.toElementType(resultElementType));

      return true;
    }

    if (lhsArrayType && rhsArrayType) {
      if (lhsArrayType.getRank() != rhsArrayType.getRank()) {
        return false;
      }

      llvm::SmallVector<int64_t> shape;

      for (const auto& [lhsDim, rhsDim] :
           llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        if (lhsDim != ArrayType::kDynamicSize) {
          shape.push_back(lhsDim);
        } else if (rhsDim != ArrayType::kDynamicSize) {
          shape.push_back(rhsDim);
        } else {
          shape.push_back(ArrayType::kDynamicSize);
        }
      }

      mlir::Type resultElementType = getMostGenericScalarType(
          lhsArrayType.getElementType(), rhsArrayType.getElementType());

      inferredTypes.push_back(ArrayType::get(shape, resultElementType));
      return true;
    }

    return false;
  }

  Results OperationLowerer::divideEW(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::divideEW>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<DivEWOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::ifelse>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    assert(operands.size() == 2);
    mlir::Type trueValueType = operands[0].getType();
    mlir::Type falseValueType = operands[1].getType();

    if (isScalarType(trueValueType) && isScalarType(falseValueType)) {
      inferredTypes.push_back(
          getMostGenericScalarType(trueValueType, falseValueType));

      return true;
    }

    auto trueValueArrayType = trueValueType.dyn_cast<ArrayType>();
    auto falseValueArrayType = falseValueType.dyn_cast<ArrayType>();

    if (trueValueArrayType && falseValueArrayType) {
      if (trueValueArrayType.getRank() != falseValueArrayType.getRank()) {
        return false;
      }

      llvm::SmallVector<int64_t> shape;

      for (const auto& [lhsDim, rhsDim] : llvm::zip(
               trueValueArrayType.getShape(),
               falseValueArrayType.getShape())) {
        if (lhsDim != ArrayType::kDynamicSize) {
          shape.push_back(lhsDim);
        } else if (rhsDim != ArrayType::kDynamicSize) {
          shape.push_back(rhsDim);
        } else {
          shape.push_back(ArrayType::kDynamicSize);
        }
      }

      mlir::Type resultElementType = getMostGenericScalarType(
          trueValueArrayType.getElementType(),
          falseValueArrayType.getElementType());

      inferredTypes.push_back(ArrayType::get(shape, resultElementType));
      return true;
    }

    return false;
  }

  Results OperationLowerer::ifElse(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    mlir::Value condition = lowerArg(*operation.getArgument(0));

    mlir::Location trueExpressionsLoc =
        loc(operation.getArgument(1)->getLocation());

    mlir::Location falseExpressionsLoc =
        loc(operation.getArgument(2)->getLocation());

    Results trueExpressions = lower(*operation.getArgument(1));
    Results falseExpressions = lower(*operation.getArgument(2));

    llvm::SmallVector<mlir::Value, 3> args;
    llvm::SmallVector<mlir::Type, 1> resultTypes;

    std::vector<mlir::Value> trueValues;
    std::vector<mlir::Value> falseValues;

    for (const auto& [trueExpression, falseExpression] :
         llvm::zip(trueExpressions, falseExpressions)) {
      mlir::Value trueValue = trueExpression.get(trueExpressionsLoc);
      mlir::Value falseValue = falseExpression.get(falseExpressionsLoc);

      trueValues.push_back(trueValue);
      falseValues.push_back(falseValue);

      args.clear();
      args.push_back(trueValue);
      args.push_back(falseValue);

      bool inferResult = inferResultTypes<ast::OperationKind::ifelse>(
          builder().getContext(), args, resultTypes);

      assert(inferResult && "Can't infer result type");
      assert(resultTypes.size() == 1);
    }

    auto selectOp = builder().create<SelectOp>(
        location, resultTypes, condition, trueValues, falseValues);

    std::vector<Reference> results;

    for (const auto& result : selectOp.getResults()) {
      results.push_back(Reference::ssa(builder(), result));
    }

    return Results(results.begin(), results.end());
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::greater>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    inferredTypes.push_back(BooleanType::get(context));
    return true;
  }

  Results OperationLowerer::greater(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::greater>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<GtOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::greaterEqual>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    inferredTypes.push_back(BooleanType::get(context));
    return true;
  }

  Results OperationLowerer::greaterOrEqual(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult =
        inferResultTypes<ast::OperationKind::greaterEqual>(
            builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<GteOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::equal>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    inferredTypes.push_back(BooleanType::get(context));
    return true;
  }

  Results OperationLowerer::equal(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::equal>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<EqOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::different>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    inferredTypes.push_back(BooleanType::get(context));
    return true;
  }

  Results OperationLowerer::notEqual(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::different>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<NotEqOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::lessEqual>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    inferredTypes.push_back(BooleanType::get(context));
    return true;
  }

  Results OperationLowerer::lessOrEqual(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::lessEqual>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<LteOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::less>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    inferredTypes.push_back(BooleanType::get(context));
    return true;
  }

  Results OperationLowerer::less(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::less>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<LtOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::land>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    assert(operands.size() == 2);
    mlir::Type lhsType = operands[0].getType();
    mlir::Type rhsType = operands[1].getType();

    if (isScalarType(lhsType) && isScalarType(rhsType)) {
      inferredTypes.push_back(BooleanType::get(context));
      return true;
    }

    auto lhsArrayType = lhsType.dyn_cast<ArrayType>();
    auto rhsArrayType = rhsType.dyn_cast<ArrayType>();

    if (lhsArrayType && rhsArrayType) {
      if (lhsArrayType.getRank() != rhsArrayType.getRank()) {
        return false;
      }

      llvm::SmallVector<int64_t> shape;

      for (const auto& [lhsDim, rhsDim] :
           llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        if (lhsDim != ArrayType::kDynamicSize) {
          shape.push_back(lhsDim);
        } else if (rhsDim != ArrayType::kDynamicSize) {
          shape.push_back(rhsDim);
        } else {
          shape.push_back(ArrayType::kDynamicSize);
        }
      }

      inferredTypes.push_back(
          ArrayType::get(shape, BooleanType::get(context)));

      return true;
    }

    return false;
  }

  Results OperationLowerer::logicalAnd(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::land>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<AndOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::lnot>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    assert(operands.size() == 1);
    mlir::Type operandType = operands[0].getType();

    if (isScalarType(operandType)) {
      inferredTypes.push_back(BooleanType::get(context));
      return true;
    }

    if (auto operandArrayType = operandType.dyn_cast<ArrayType>()) {
      inferredTypes.push_back(
          operandArrayType.toElementType(BooleanType::get(context)));

      return true;
    }

    return false;
  }

  Results OperationLowerer::logicalNot(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 1);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::lnot>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<NotOp>(
        location, resultTypes[0], args[0]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::lor>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    assert(operands.size() == 2);
    mlir::Type lhsType = operands[0].getType();
    mlir::Type rhsType = operands[1].getType();

    if (isScalarType(lhsType) && isScalarType(rhsType)) {
      inferredTypes.push_back(BooleanType::get(context));
      return true;
    }

    auto lhsArrayType = lhsType.dyn_cast<ArrayType>();
    auto rhsArrayType = rhsType.dyn_cast<ArrayType>();

    if (lhsArrayType && rhsArrayType) {
      if (lhsArrayType.getRank() != rhsArrayType.getRank()) {
        return false;
      }

      llvm::SmallVector<int64_t> shape;

      for (const auto& [lhsDim, rhsDim] :
           llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        if (lhsDim != ArrayType::kDynamicSize) {
          shape.push_back(lhsDim);
        } else if (rhsDim != ArrayType::kDynamicSize) {
          shape.push_back(rhsDim);
        } else {
          shape.push_back(ArrayType::kDynamicSize);
        }
      }

      inferredTypes.push_back(
          ArrayType::get(shape, BooleanType::get(context)));

      return true;
    }

    return false;
  }

  Results OperationLowerer::logicalOr(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::lor>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<OrOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  Results OperationLowerer::subscription(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 4> args;
    lowerArgs(operation, args);
    assert(args.size() >= 1);

    mlir::Value array = args[0];
    assert(array.getType().isa<ArrayType>());

    // Indices in Modelica are 1-based, while in the MLIR dialect are
    // 0-based. Thus, we need to shift them by one. In doing so, we also
    // force the result to be of index type.
    std::vector<mlir::Value> zeroBasedIndices;

    for (size_t i = 1; i < args.size(); ++i) {
      mlir::Value index = args[i];

      mlir::Value one = builder().create<ConstantOp>(
          index.getLoc(), builder().getIndexAttr(-1));

      mlir::Value zeroBasedIndex = builder().create<AddOp>(
          index.getLoc(), builder().getIndexType(), index, one);

      zeroBasedIndices.push_back(zeroBasedIndex);
    }

    mlir::Value result = builder().create<SubscriptionOp>(
        location, array, zeroBasedIndices);

    return Reference::memory(builder(), result);
  }

  Results OperationLowerer::memberLookup(const ast::Operation& operation)
  {
    llvm_unreachable("Member lookup is not implemented yet.");
    return Results();
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::powerOf>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    assert(operands.size() == 2);
    mlir::Type baseType = operands[0].getType();
    mlir::Type exponentType = operands[1].getType();

    auto inferResultType =
        [](mlir::Type base, mlir::Type exponent) -> mlir::Type {
      if (exponent.isa<RealType>()) {
        return exponent;
      }

      return base;
    };

    if (isScalarType(baseType)) {
      inferredTypes.push_back(inferResultType(baseType, exponentType));
      return true;
    }

    if (auto baseArrayType = baseType.dyn_cast<ArrayType>()) {
      inferredTypes.push_back(baseArrayType);
      return true;
    }

    return false;
  }

  Results OperationLowerer::powerOf(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::powerOf>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<PowOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }

  template<>
  bool OperationLowerer::inferResultTypes<ast::OperationKind::powerOfEW>(
      mlir::MLIRContext* context,
      llvm::ArrayRef<mlir::Value> operands,
      llvm::SmallVectorImpl<mlir::Type>& inferredTypes)
  {
    assert(operands.size() == 2);
    mlir::Type baseType = operands[0].getType();
    mlir::Type exponentType = operands[1].getType();

    auto inferResultType =
        [](mlir::Type base, mlir::Type exponent) -> mlir::Type {
          if (exponent.isa<RealType>()) {
            return exponent;
          }

          return base;
        };

    if (isScalarType(baseType) && isScalarType(exponentType)) {
      inferredTypes.push_back(inferResultType(baseType, exponentType));
      return true;
    }

    auto baseArrayType = baseType.dyn_cast<ArrayType>();
    auto exponentArrayType = exponentType.dyn_cast<ArrayType>();

    if (isScalarType(baseType) && exponentArrayType) {
      inferredTypes.push_back(exponentArrayType.toElementType(
          inferResultType(baseType, exponentArrayType.getElementType())));

      return true;
    }

    if (baseArrayType && isScalarType(exponentType)) {
      inferredTypes.push_back(baseArrayType.toElementType(
          inferResultType(baseArrayType.getElementType(), exponentType)));
      return true;
    }

    if (baseArrayType && exponentArrayType) {
      if (baseArrayType.getRank() != exponentArrayType.getRank()) {
        return false;
      }

      llvm::SmallVector<int64_t> shape;

      for (const auto& [lhsDim, rhsDim] :
           llvm::zip(baseArrayType.getShape(), exponentArrayType.getShape())) {
        if (lhsDim != ArrayType::kDynamicSize) {
          shape.push_back(lhsDim);
        } else if (rhsDim != ArrayType::kDynamicSize) {
          shape.push_back(rhsDim);
        } else {
          shape.push_back(ArrayType::kDynamicSize);
        }
      }

      mlir::Type resultElementType = inferResultType(
          baseArrayType.getElementType(), exponentArrayType.getElementType());

      inferredTypes.push_back(ArrayType::get(shape, resultElementType));
      return true;
    }

    return false;
  }

  Results OperationLowerer::powerOfEW(const ast::Operation& operation)
  {
    mlir::Location location = loc(operation.getLocation());

    llvm::SmallVector<mlir::Value, 2> args;
    lowerArgs(operation, args);
    assert(args.size() == 2);

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    bool inferResult = inferResultTypes<ast::OperationKind::powerOfEW>(
        builder().getContext(), args, resultTypes);

    assert(inferResult && "Can't infer result type");
    assert(resultTypes.size() == 1);

    mlir::Value result = builder().create<PowEWOp>(
        location, resultTypes[0], args[0], args[1]);

    return Reference::ssa(builder(), result);
  }
}
