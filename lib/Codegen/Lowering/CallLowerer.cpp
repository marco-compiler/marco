#include "marco/Codegen/Lowering/CallLowerer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  CallLowerer::CallLowerer(LoweringContext* context, BridgeInterface* bridge)
    : Lowerer(context, bridge)
  {
  }

  Results CallLowerer::userDefinedFunction(const Call& call)
  {
    std::vector<Reference> results;
    std::vector<mlir::Value> args;

    for (const auto& arg : call) {
      auto reference = lower(*arg)[0];
      args.push_back(*reference);
    }

    auto resultType = call.getType();
    std::vector<mlir::Type> resultsTypes;

    if (resultType.isa<PackedType>()) {
      for (const auto& type : resultType.get<PackedType>()) {
        resultsTypes.push_back(lower(type));
      }
    } else {
      resultsTypes.push_back(lower(resultType));
    }

    auto op = builder().create<CallOp>(
        loc(call.getLocation()),
        call.getFunction()->get<ReferenceAccess>()->getName(),
        resultsTypes, args);

    for (auto result : op->getResults()) {
      results.push_back(Reference::ssa(&builder(), result));
    }

    return Results(results.begin(), results.end());
  }

  Results CallLowerer::abs(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "abs");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<AbsOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::acos(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "acos");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<AcosOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::asin(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "asin");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<AsinOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::atan(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "atan");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<AtanOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::atan2(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "atan2");
    assert(call.argumentsCount() == 2);

    auto location = loc(call.getLocation());

    mlir::Value y = *lower(*call.getArg(0))[0];
    mlir::Value x = *lower(*call.getArg(1))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<Atan2Op>(location, resultType, y, x);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::cos(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "cos");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<CosOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::cosh(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "cosh");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<CoshOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::der(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "der");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<DerOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::diagonal(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "diagonal");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<DiagonalOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::exp(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "exp");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<ExpOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::identity(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "identity");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<IdentityOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::linspace(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "linspace");
    assert(call.argumentsCount() == 3);

    auto location = loc(call.getLocation());

    mlir::Value start = *lower(*call.getArg(0))[0];
    mlir::Value end = *lower(*call.getArg(1))[0];
    mlir::Value steps = *lower(*call.getArg(2))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<LinspaceOp>(location, resultType, start, end, steps);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::log(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "log");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<LogOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::log10(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "log10");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<Log10Op>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::max(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "max");
    assert(call.argumentsCount() == 1 || call.argumentsCount() == 2);

    auto location = loc(call.getLocation());

    std::vector<mlir::Value> args;

    for (const auto& arg : call) {
      args.push_back(*lower(*arg)[0]);
    }

    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<MaxOp>(location, resultType, args);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::min(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "min");
    assert(call.argumentsCount() == 1 || call.argumentsCount() == 2);

    auto location = loc(call.getLocation());

    std::vector<mlir::Value> args;

    for (const auto& arg : call) {
      args.push_back(*lower(*arg)[0]);
    }

    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<MinOp>(location, resultType, args);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::mod(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "mod");
    assert(call.argumentsCount() == 2);

    auto location = loc(call.getLocation());

    mlir::Value dividend = *lower(*call.getArg(0))[0];
    mlir::Value divisor = *lower(*call.getArg(1))[0];

    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<ModOp>(location, resultType, dividend, divisor);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::ndims(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "ndims");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value array = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<NDimsOp>(location, resultType, array);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::ones(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "ones");

    auto location = loc(call.getLocation());
    mlir::Type resultType = lower(call.getType());

    // The number of operands is equal to the rank of the resulting array
    assert(call.argumentsCount() == resultType.cast<ArrayType>().getRank());

    std::vector<mlir::Value> dimensions;

    for (const auto& arg : call) {
      dimensions.push_back(*lower(*arg)[0]);
    }

    mlir::Value result = builder().create<OnesOp>(location, resultType, dimensions);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::product(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "product");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value array = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<ProductOp>(location, resultType, array);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::sign(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "sign");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value array = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<SignOp>(location, resultType, array);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::sin(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "sin");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<SinOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::sinh(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "sinh");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<SinhOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::size(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "size");
    assert(call.argumentsCount() == 1 || call.argumentsCount() == 2);

    auto location = loc(call.getLocation());

    std::vector<Reference> results;
    std::vector<mlir::Value> args;

    for (const auto& arg : call) {
      args.push_back(*lower(*arg)[0]);
    }

    mlir::Type resultType = lower(call.getType());

    if (args.size() == 1) {
      mlir::Value result = builder().create<SizeOp>(location, resultType, args);
      return Reference::ssa(&builder(), result);
    }

    if (args.size() == 2) {
      mlir::Value oneValue = builder().create<ConstantOp>(location, IntegerAttr::get(builder().getContext(), 1));
      mlir::Value index = builder().create<SubOp>(location, builder().getIndexType(), args[1], oneValue);
      mlir::Value result = builder().create<SizeOp>(location, resultType, args[0], index);
      return Reference::ssa(&builder(), result);
    }

    llvm_unreachable("Unexpected number of arguments for 'size' function call");
    return Results();
  }

  Results CallLowerer::sqrt(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "sqrt");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<SqrtOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::sum(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "sum");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value array = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<SumOp>(location, resultType, array);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::symmetric(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "symmetric");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value array = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<SymmetricOp>(location, resultType, array);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::tan(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "tan");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<TanOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::tanh(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "tanh");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value operand = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<TanhOp>(location, resultType, operand);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::transpose(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "transpose");
    assert(call.argumentsCount() == 1);

    auto location = loc(call.getLocation());

    mlir::Value array = *lower(*call.getArg(0))[0];
    mlir::Type resultType = lower(call.getType());
    mlir::Value result = builder().create<TransposeOp>(location, resultType, array);
    return Reference::ssa(&builder(), result);
  }

  Results CallLowerer::zeros(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "zeros");

    auto location = loc(call.getLocation());
    mlir::Type resultType = lower(call.getType());

    // The number of operands is equal to the rank of the resulting array
    assert(call.argumentsCount() == resultType.cast<ArrayType>().getRank());

    std::vector<mlir::Value> dimensions;

    for (const auto& arg : call) {
      dimensions.push_back(*lower(*arg)[0]);
    }

    mlir::Value result = builder().create<ZerosOp>(location, resultType, dimensions);
    return Reference::ssa(&builder(), result);
  }
}
