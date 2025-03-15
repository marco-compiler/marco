#include "marco/Codegen/Lowering/OperationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen::lowering;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering {
OperationLowerer::OperationLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

std::optional<Results>
OperationLowerer::lower(const ast::Operation &operation) {
  auto lowererFn =
      [](ast::OperationKind kind) -> OperationLowerer::LoweringFunction {
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

    case ast::OperationKind::powerOf:
      return &OperationLowerer::powerOf;

    case ast::OperationKind::powerOfEW:
      return &OperationLowerer::powerOfEW;

    case ast::OperationKind::range:
      return &OperationLowerer::range;

    default:
      return nullptr;
    }
  };

  auto lowerer = lowererFn(operation.getOperationKind());
  assert(lowerer != nullptr && "Unknown operation type");
  return lowerer(*this, operation);
}

std::optional<mlir::Value>
OperationLowerer::lowerArg(const ast::Expression &expression) {
  mlir::Location location = loc(expression.getLocation());
  auto loweredExpression = lower(expression);
  if (!loweredExpression) {
    return std::nullopt;
  }
  auto &results = *loweredExpression;
  assert(results.size() == 1);
  return results[0].get(location);
}

bool OperationLowerer::lowerArgs(const ast::Operation &operation,
                                 llvm::SmallVectorImpl<mlir::Value> &args) {
  for (size_t i = 0, e = operation.getNumOfArguments(); i < e; ++i) {
    auto arg = lowerArg(*operation.getArgument(i));
    if (!arg) {
      return false;
    }
    args.push_back(*arg);
  }
  return true;
}

std::optional<Results>
OperationLowerer::negate(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 1> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 1);

  mlir::Value result = builder().create<NegateOp>(location, args[0]);
  return Reference::ssa(builder(), result);
}

std::optional<Results> OperationLowerer::add(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() >= 2);

  llvm::SmallVector<mlir::Value, 2> current;
  current.push_back(args[0]);
  current.push_back(args[1]);

  mlir::Value result =
      builder().create<AddOp>(location, current[0], current[1]);

  for (size_t i = 2; i < args.size(); ++i) {
    current.clear();
    args.push_back(result);
    args.push_back(args[i]);
    result = builder().create<AddOp>(location, current[0], current[1]);
  }

  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::addEW(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<AddEWOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::subtract(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<SubOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::subtractEW(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<SubEWOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::multiply(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() >= 2);

  llvm::SmallVector<mlir::Value, 2> current;
  current.push_back(args[0]);
  current.push_back(args[1]);

  mlir::Value result =
      builder().create<MulOp>(location, current[0], current[1]);

  for (size_t i = 2; i < args.size(); ++i) {
    current.clear();
    args.push_back(result);
    args.push_back(args[i]);

    result = builder().create<MulOp>(location, current[0], current[1]);
  }

  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::multiplyEW(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<MulEWOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::divide(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<DivOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::divideEW(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<DivEWOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::ifElse(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  std::optional<mlir::Value> condition = lowerArg(*operation.getArgument(0));
  if (!condition) {
    return std::nullopt;
  }

  mlir::Location trueExpressionsLoc =
      loc(operation.getArgument(1)->getLocation());

  mlir::Location falseExpressionsLoc =
      loc(operation.getArgument(2)->getLocation());

  std::optional<Results> trueExpressions = lower(*operation.getArgument(1));
  if (!trueExpressions) {
    return std::nullopt;
  }
  std::optional<Results> falseExpressions = lower(*operation.getArgument(2));
  if (!falseExpressions) {
    return std::nullopt;
  }

  llvm::SmallVector<mlir::Value, 3> args;

  std::vector<mlir::Value> trueValues;
  std::vector<mlir::Value> falseValues;

  for (const auto &[trueExpression, falseExpression] :
       llvm::zip(*trueExpressions, *falseExpressions)) {
    mlir::Value trueValue = trueExpression.get(trueExpressionsLoc);
    mlir::Value falseValue = falseExpression.get(falseExpressionsLoc);

    trueValues.push_back(trueValue);
    falseValues.push_back(falseValue);

    args.clear();
    args.push_back(trueValue);
    args.push_back(falseValue);
  }

  auto selectOp =
      builder().create<SelectOp>(location, *condition, trueValues, falseValues);

  std::vector<Reference> results;

  for (const auto &result : selectOp.getResults()) {
    results.push_back(Reference::ssa(builder(), result));
  }

  return Results({results.begin(), results.end()});
}

std::optional<Results>
OperationLowerer::greater(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<GtOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::greaterOrEqual(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<GteOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::equal(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<EqOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::notEqual(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<NotEqOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::lessOrEqual(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<LteOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results> OperationLowerer::less(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<LtOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::logicalAnd(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<AndOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::logicalNot(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 1);

  mlir::Value result = builder().create<NotOp>(location, args[0]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::logicalOr(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<OrOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::subscription(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());
  llvm::SmallVector<mlir::Value, 4> args;

  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }

  assert(!args.empty());
  mlir::Value array = args[0];
  assert(mlir::isa<mlir::ShapedType>(array.getType()));

  llvm::SmallVector<mlir::Value> indices;

  for (size_t i = 1, e = args.size(); i < e; ++i) {
    indices.push_back(args[i]);
  }

  mlir::Value result = builder().create<TensorViewOp>(location, array, indices);
  return Reference::tensor(builder(), result);
}

std::optional<Results>
OperationLowerer::powerOf(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<PowOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::powerOfEW(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 2> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2);

  mlir::Value result = builder().create<PowEWOp>(location, args[0], args[1]);
  return Reference::ssa(builder(), result);
}

std::optional<Results>
OperationLowerer::range(const ast::Operation &operation) {
  mlir::Location location = loc(operation.getLocation());

  llvm::SmallVector<mlir::Value, 3> args;
  if (!lowerArgs(operation, args)) {
    return std::nullopt;
  }
  assert(args.size() == 2 || args.size() == 3);

  if (args.size() == 2) {
    args.push_back(args[1]);

    mlir::Value step =
        builder().create<ConstantOp>(location, builder().getIndexAttr(1));

    args[1] = step;
  }

  mlir::Value result =
      builder().create<RangeOp>(location, args[0], args[2], args[1]);

  return Reference::ssa(builder(), result);
}
} // namespace marco::codegen::lowering
