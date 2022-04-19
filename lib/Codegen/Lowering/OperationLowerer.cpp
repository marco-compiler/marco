#include "marco/Codegen/Lowering/OperationLowerer.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  OperationLowerer::OperationLowerer(LoweringContext* context, BridgeInterface* bridge)
      : Lowerer(context, bridge)
  {
  }

  std::vector<mlir::Value> OperationLowerer::lowerArgs(const Operation& operation)
  {
    std::vector<mlir::Value> args;

    for (const auto& arg : operation) {
      args.push_back(*lower(*arg)[0]);
    }

    return args;
  }

  Results OperationLowerer::negate(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::negate, 1>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      mlir::Value result = builder().create<NotOp>(loc, resultType, args[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::add(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::add, -1>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      mlir::Value result = builder().create<AddOp>(loc, resultType, args[0], args[1]);

      for (size_t i = 2; i < args.size(); ++i) {
        result = builder().create<AddOp>(loc, resultType, result, args[i]);
      }

      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::subtract(const ast::Operation& operation)
  {
    assert(operation.getOperationKind() == OperationKind::subtract);

    mlir::Type resultType = lower(operation.getType());

    if (operation.getArguments().size() == 1) {
      // TODO
      // Special case for sign change (i.e "-x").
      // In future, when all the project will rely on MLIR, a different
      // operation in the frontend should be created for this purpose.

      return lowerOperation<OperationKind::subtract, 1>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
        mlir::Value result = builder().create<NegateOp>(loc, resultType, args[0]);
        return Reference::ssa(&builder(), result);
      });
    }

    if (operation.getArguments().size() == 2) {
      return lowerOperation<OperationKind::subtract, 2>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
        mlir::Value result = builder().create<SubOp>(loc, resultType, args[0], args[1]);
        return Reference::ssa(&builder(), result);
      });
    }

    llvm_unreachable("Unexpect number of arguments for subtract operation.");
    return Results();
  }

  Results OperationLowerer::multiply(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::multiply, -1>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      mlir::Value result = builder().create<MulOp>(loc, resultType, args[0], args[1]);

      for (size_t i = 2; i < args.size(); ++i) {
        result = builder().create<MulOp>(loc, resultType, result, args[i]);
      }

      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::divide(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::divide, 2>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      mlir::Value result = builder().create<DivOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::ifElse(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::ifelse, 3>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      mlir::Value trueValue = builder().create<CastOp>(args[1].getLoc(), resultType, args[1]);
      mlir::Value falseValue = builder().create<CastOp>(args[2].getLoc(), resultType, args[2]);

      mlir::Value result = builder().create<mlir::SelectOp>(loc, args[0], trueValue, falseValue);
      result = builder().create<CastOp>(result.getLoc(), resultType, result);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::greater(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::greater, 2>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      mlir::Value result = builder().create<GtOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::greaterOrEqual(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::greaterEqual, 2>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      mlir::Value result = builder().create<GteOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::equal(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::equal, 2>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      mlir::Value result = builder().create<EqOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::notEqual(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::different, 2>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      mlir::Value result = builder().create<NotEqOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::lessOrEqual(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::lessEqual, 2>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      mlir::Value result = builder().create<LteOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::less(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::less, 2>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      mlir::Value result = builder().create<LtOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::logicalAnd(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::land, 2>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      mlir::Value result = builder().create<AndOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::logicalOr(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::lor, 2>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      mlir::Value result = builder().create<OrOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::subscription(const ast::Operation& operation)
  {
    return lowerOperation<OperationKind::subscription>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args[0].getType().isa<ArrayType>());
      mlir::Value result = builder().create<SubscriptionOp>(loc, args[0], args.drop_front());
      return Reference::memory(&builder(), result);
    });
  }

  Results OperationLowerer::memberLookup(const ast::Operation& operation)
  {
    llvm_unreachable("Member lookup is not implemented yet.");
    return Results();
  }

  Results OperationLowerer::powerOf(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::powerOf, 2>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      mlir::Value base = args[0];
      mlir::Value exponent = args[1];

      mlir::Value result = builder().create<PowOp>(loc, resultType, base, exponent);
      return Reference::ssa(&builder(), result);
    });
  }
}
