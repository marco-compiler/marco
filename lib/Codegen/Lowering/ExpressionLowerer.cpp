#include "marco/Codegen/Lowering/ExpressionLowerer.h"
#include "marco/Codegen/Lowering/CallLowerer.h"
#include "marco/Codegen/Lowering/OperationLowerer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  ExpressionLowerer::ExpressionLowerer(LoweringContext* context, BridgeInterface* bridge)
      : Lowerer(context, bridge),
        callLowerer(std::make_unique<CallLowerer>(context, bridge)),
        operationLowerer(std::make_unique<OperationLowerer>(context, bridge))
  {
  }

  Results ExpressionLowerer::operator()(const Array& array)
  {
    mlir::Location location = loc(array.getLocation());
    auto arrayType = lower(array.getType()).cast<ArrayType>();

    mlir::Value result = builder().create<AllocOp>(location, arrayType, llvm::None);

    for (const auto& value : llvm::enumerate(array)) {
      mlir::Value index = builder().create<ConstantOp>(location, builder().getIndexAttr(value.index()));
      mlir::Value slice = builder().create<SubscriptionOp>(location, result, index);
      builder().create<AssignmentOp>(location, slice, *lower(*value.value())[0]);
    }

    return Reference::ssa(&builder(), result);
  }

  Results ExpressionLowerer::operator()(const Call& call)
  {
    const auto* function = call.getFunction();
    const auto& functionName = function->get<ReferenceAccess>()->getName();

    auto lowerer = llvm::StringSwitch<CallLowerer::LoweringFunction>(functionName)
      .Case("abs", &CallLowerer::abs)
      .Case("acos", &CallLowerer::acos)
      .Case("asin", &CallLowerer::asin)
      .Case("atan", &CallLowerer::atan)
      .Case("atan2", &CallLowerer::atan2)
      .Case("cos", &CallLowerer::cos)
      .Case("cosh", &CallLowerer::cosh)
      .Case("der", &CallLowerer::der)
      .Case("diagonal", &CallLowerer::diagonal)
      .Case("exp", &CallLowerer::exp)
      .Case("floor", &CallLowerer::floor)
      .Case("identity", &CallLowerer::identity)
      .Case("linspace", &CallLowerer::linspace)
      .Case("log", &CallLowerer::log)
      .Case("log10", &CallLowerer::log10)
      .Case("max", &CallLowerer::max)
      .Case("min", &CallLowerer::min)
      .Case("mod", &CallLowerer::mod)
      .Case("ndims", &CallLowerer::ndims)
      .Case("ones", &CallLowerer::ones)
      .Case("product", &CallLowerer::product)
      .Case("sign", &CallLowerer::sign)
      .Case("sin", &CallLowerer::sin)
      .Case("sinh", &CallLowerer::sinh)
      .Case("size", &CallLowerer::size)
      .Case("sqrt", &CallLowerer::sqrt)
      .Case("sum", &CallLowerer::sum)
      .Case("symmetric", &CallLowerer::symmetric)
      .Case("tan", &CallLowerer::tan)
      .Case("tanh", &CallLowerer::tanh)
      .Case("transpose", &CallLowerer::transpose)
      .Case("zeros", &CallLowerer::zeros)
      .Default(&CallLowerer::userDefinedFunction);

    assert(callLowerer != nullptr);
    return lowerer(*callLowerer, call);
  }

  Results ExpressionLowerer::operator()(const Constant& constant)
  {
    const auto& type = constant.getType();

    assert(type.isa<BuiltInType>() && "Constants can be made only of built-in typed values");
    auto builtInType = type.get<BuiltInType>();

    mlir::Attribute attribute;

    if (builtInType == BuiltInType::Boolean) {
      attribute = BooleanAttr::get(builder().getContext(), constant.as<BuiltInType::Boolean>());
    } else if (builtInType == BuiltInType::Integer) {
      attribute = IntegerAttr::get(builder().getContext(), constant.as<BuiltInType::Integer>());
    } else if (builtInType == BuiltInType::Real) {
      attribute = RealAttr::get(builder().getContext(), constant.as<BuiltInType::Real>());
    } else {
      llvm_unreachable("Unsupported constant type");
    }

    auto result = builder().create<ConstantOp>(loc(constant.getLocation()), attribute);
    return Reference::ssa(&builder(), result);
  }

  Results ExpressionLowerer::operator()(const Operation& operation)
  {
    auto lowererFn = [](OperationKind kind) -> OperationLowerer::LoweringFunction {
      switch (kind) {
        case OperationKind::negate:
          return &OperationLowerer::negate;

        case OperationKind::add:
          return &OperationLowerer::add;

        case OperationKind::addEW:
          return &OperationLowerer::addEW;

        case OperationKind::subtract:
          return &OperationLowerer::subtract;

        case OperationKind::subtractEW:
          return &OperationLowerer::subtractEW;

        case OperationKind::multiply:
          return &OperationLowerer::multiply;

        case OperationKind::multiplyEW:
          return &OperationLowerer::multiplyEW;

        case OperationKind::divide:
          return &OperationLowerer::divide;

        case OperationKind::divideEW:
          return &OperationLowerer::divideEW;

        case OperationKind::ifelse:
          return &OperationLowerer::ifElse;

        case OperationKind::greater:
          return &OperationLowerer::greater;

        case OperationKind::greaterEqual:
          return &OperationLowerer::greaterOrEqual;

        case OperationKind::equal:
          return &OperationLowerer::equal;

        case OperationKind::different:
          return &OperationLowerer::notEqual;

        case OperationKind::lessEqual:
          return &OperationLowerer::lessOrEqual;

        case OperationKind::less:
          return &OperationLowerer::less;

        case OperationKind::land:
          return &OperationLowerer::logicalAnd;

        case OperationKind::lnot:
          return &OperationLowerer::logicalNot;

        case OperationKind::lor:
          return &OperationLowerer::logicalOr;

        case OperationKind::subscription:
          return &OperationLowerer::subscription;

        case OperationKind::memberLookup:
          return &OperationLowerer::memberLookup;

        case OperationKind::powerOf:
          return &OperationLowerer::powerOf;

        case OperationKind::powerOfEW:
          return &OperationLowerer::powerOfEW;
      }

      llvm_unreachable("Unknown operation type");
      return nullptr;
    };

    auto lowerer = lowererFn(operation.getOperationKind());
    assert(operationLowerer != nullptr);
    return lowerer(*operationLowerer, operation);
  }

  Results ExpressionLowerer::operator()(const ReferenceAccess& reference)
  {
    return symbolTable().lookup(reference.getName());
  }

  Results ExpressionLowerer::operator()(const Tuple& tuple)
  {
    Results result;

    for (const auto& exp : tuple) {
      auto values = lower(*exp);

      // The only way to have multiple returns is to call a function, but
      // this is forbidden in a tuple declaration. In fact, a tuple is just
      // a container of references.
      assert(values.size() == 1);
      result.append(values[0]);
    }

    return result;
  }

  Results ExpressionLowerer::operator()(const RecordInstance& record)
  {
    llvm_unreachable("Not implemented");
    return Results();
  }
}
