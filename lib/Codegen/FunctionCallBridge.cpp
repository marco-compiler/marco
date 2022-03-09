#include "marco/Codegen/FunctionCallBridge.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  FunctionCallBridge::FunctionCallBridge(NewLoweringBridge* bridge)
    : bridge(bridge)
  {
  }

  FunctionCallBridge::Results FunctionCallBridge::userDefinedFunction(const Call& call)
  {
    std::vector<Reference> results;
    std::vector<mlir::Value> args;

    for (const auto& arg : call) {
      auto reference = bridge->lower<Expression>(*arg)[0];
      args.push_back(*reference);
    }

    auto resultType = call.getType();
    std::vector<mlir::Type> resultsTypes;

    if (resultType.isa<PackedType>()) {
      for (const auto& type : resultType.get<PackedType>()) {
        resultsTypes.push_back(bridge->lower(type, ArrayAllocationScope::heap));
      }
    } else {
      resultsTypes.push_back(bridge->lower(resultType, ArrayAllocationScope::heap));
    }

    auto op = bridge->builder.create<CallOp>(
        bridge->loc(call.getLocation()),
        call.getFunction()->get<ReferenceAccess>()->getName(),
        resultsTypes, args);

    for (auto result : op->getResults()) {
      results.push_back(Reference::ssa(&bridge->builder, result));
    }

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::abs(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "abs");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<AbsOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::acos(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "acos");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<AcosOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::asin(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "asin");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<AsinOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::atan(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "atan");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<AtanOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::atan2(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "atan2");
    assert(call.argumentsCount() == 2);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value y = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Value x = *bridge->lower<Expression>(*call.getArg(1))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<Atan2Op>(loc, resultType, y, x);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::cos(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "cos");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<CosOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::cosh(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "cosh");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<CoshOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::der(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "der");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    assert(operand.getType().isa<ArrayType>());
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<DerOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::diagonal(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "diagonal");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<DiagonalOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::exp(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "exp");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<ExpOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::identity(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "identity");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<IdentityOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::linspace(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "linspace");
    assert(call.argumentsCount() == 3);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value start = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Value end = *bridge->lower<Expression>(*call.getArg(1))[0];
    mlir::Value steps = *bridge->lower<Expression>(*call.getArg(2))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<LinspaceOp>(loc, resultType, start, end, steps);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::log(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "log");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<LogOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::log10(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "log10");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<Log10Op>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::max(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "max");
    assert(call.argumentsCount() == 1 || call.argumentsCount() == 2);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    std::vector<mlir::Value> args;

    for (const auto& arg : call) {
      args.push_back(*bridge->lower<Expression>(*arg)[0]);
    }

    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<MaxOp>(loc, resultType, args);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::min(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "min");
    assert(call.argumentsCount() == 1 || call.argumentsCount() == 2);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    std::vector<mlir::Value> args;

    for (const auto& arg : call) {
      args.push_back(*bridge->lower<Expression>(*arg)[0]);
    }

    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<MinOp>(loc, resultType, args);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::ndims(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "ndims");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value array = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<NDimsOp>(loc, resultType, array);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::ones(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "ones");

    auto loc = bridge->loc(call.getLocation());
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);

    // The number of operands is equal to the rank of the resulting array
    assert(call.argumentsCount() == resultType.cast<ArrayType>().getRank());

    std::vector<Reference> results;
    std::vector<mlir::Value> dimensions;

    for (const auto& arg : call) {
      dimensions.push_back(*bridge->lower<Expression>(*arg)[0]);
    }

    mlir::Value result = bridge->builder.create<OnesOp>(loc, resultType, dimensions);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::product(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "product");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value array = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<ProductOp>(loc, resultType, array);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::sign(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "sign");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value array = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<SignOp>(loc, resultType, array);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::sin(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "sin");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<SinOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::sinh(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "sinh");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<SinOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::size(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "size");
    assert(call.argumentsCount() == 1 || call.argumentsCount() == 2);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    std::vector<mlir::Value> args;

    for (const auto& arg : call) {
      args.push_back(*bridge->lower<Expression>(*arg)[0]);
    }

    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);

    if (args.size() == 1) {
      mlir::Value result = bridge->builder.create<SizeOp>(loc, resultType, args);
      results.push_back(Reference::ssa(&bridge->builder, result));
    } else if (args.size() == 2) {
      mlir::Value oneValue = bridge->builder.create<ConstantOp>(loc, bridge->builder.getIntegerAttribute(1));
      mlir::Value index = bridge->builder.create<SubOp>(loc,bridge->builder.getIndexType(), args[1], oneValue);
      mlir::Value result = bridge->builder.create<SizeOp>(loc, resultType, args[0], index);
      results.emplace_back(Reference::ssa(&bridge->builder, result));
    }

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::sqrt(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "sqrt");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<SqrtOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::sum(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "sum");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value array = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<SumOp>(loc, resultType, array);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::symmetric(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "symmetric");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value array = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<SymmetricOp>(loc, resultType, array);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::tan(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "tan");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<TanOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::tanh(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "tanh");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value operand = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<TanhOp>(loc, resultType, operand);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::transpose(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "transpose");
    assert(call.argumentsCount() == 1);

    auto loc = bridge->loc(call.getLocation());

    std::vector<Reference> results;
    mlir::Value array = *bridge->lower<Expression>(*call.getArg(0))[0];
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);
    mlir::Value result = bridge->builder.create<TransposeOp>(loc, resultType, array);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }

  FunctionCallBridge::Results FunctionCallBridge::zeros(const Call& call)
  {
    assert(call.getFunction()->get<ReferenceAccess>()->getName() == "ones");

    auto loc = bridge->loc(call.getLocation());
    mlir::Type resultType = bridge->lower(call.getType(), ArrayAllocationScope::stack);

    // The number of operands is equal to the rank of the resulting array
    assert(call.argumentsCount() == resultType.cast<ArrayType>().getRank());

    std::vector<Reference> results;
    std::vector<mlir::Value> dimensions;

    for (const auto& arg : call) {
      dimensions.push_back(*bridge->lower<Expression>(*arg)[0]);
    }

    mlir::Value result = bridge->builder.create<ZerosOp>(loc, resultType, dimensions);
    results.push_back(Reference::ssa(&bridge->builder, result));

    return results;
  }
}
