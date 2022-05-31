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

    return lowerOperation<OperationKind::negate>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 1);
      mlir::Value result = builder().create<NotOp>(loc, resultType, args[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::add(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::add>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() >= 2);
      mlir::Value result = builder().create<AddOp>(loc, resultType, args[0], args[1]);

      for (size_t i = 2; i < args.size(); ++i) {
        result = builder().create<AddOp>(loc, resultType, result, args[i]);
      }

      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::addEW(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::addEW>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 2);
      mlir::Value result = builder().create<AddEWOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::subtract(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    if (operation.getArguments().size() == 1) {
      // TODO
      // Special case for sign change (i.e "-x").
      // In future, when all the project will rely on MLIR, a different
      // operation in the frontend should be created for this purpose.

      return lowerOperation<OperationKind::subtract>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
        assert(args.size() == 1);
        mlir::Value result = builder().create<NegateOp>(loc, resultType, args[0]);
        return Reference::ssa(&builder(), result);
      });
    }

    if (operation.getArguments().size() == 2) {
      return lowerOperation<OperationKind::subtract>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
        assert(args.size() == 2);
        mlir::Value result = builder().create<SubOp>(loc, resultType, args[0], args[1]);
        return Reference::ssa(&builder(), result);
      });
    }

    llvm_unreachable("Unexpect number of arguments for subtract operation.");
    return Results();
  }

  Results OperationLowerer::subtractEW(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::subtractEW>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 2);
      mlir::Value result = builder().create<SubEWOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::multiply(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::multiply>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() >= 2);
      mlir::Value result = builder().create<MulOp>(loc, resultType, args[0], args[1]);

      for (size_t i = 2; i < args.size(); ++i) {
        result = builder().create<MulOp>(loc, resultType, result, args[i]);
      }

      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::multiplyEW(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::multiplyEW>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 2);
      mlir::Value result = builder().create<MulEWOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::divide(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::divide>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 2);
      mlir::Value result = builder().create<DivOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::divideEW(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::divideEW>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 2);
      mlir::Value result = builder().create<DivEWOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::ifElse(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::ifelse>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 3);
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

    return lowerOperation<OperationKind::greater>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 2);
      mlir::Value result = builder().create<GtOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::greaterOrEqual(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::greaterEqual>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 2);
      mlir::Value result = builder().create<GteOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::equal(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::equal>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 2);
      mlir::Value result = builder().create<EqOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::notEqual(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::different>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 2);
      mlir::Value result = builder().create<NotEqOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::lessOrEqual(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::lessEqual>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 2);
      mlir::Value result = builder().create<LteOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::less(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::less>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 2);
      mlir::Value result = builder().create<LtOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::logicalAnd(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::land>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 2);
      mlir::Value result = builder().create<AndOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::logicalOr(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::lor>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 2);
      mlir::Value result = builder().create<OrOp>(loc, resultType, args[0], args[1]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::subscription(const ast::Operation& operation)
  {
    return lowerOperation<OperationKind::subscription>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() >= 1);
      assert(args[0].getType().isa<ArrayType>());

      // Indices in Modelica are 1-based, while in the MLIR dialect are 0-based.
      // Thus, we need to shift them by one.
      std::vector<mlir::Value> zeroBasedIndices;

      for (const auto& index : args.drop_front()) {
        mlir::Value one = builder().create<ConstantOp>(index.getLoc(), builder().getIndexAttr(-1));
        mlir::Value zeroBasedIndex = builder().create<AddOp>(index.getLoc(), index.getType(), index, one);
        zeroBasedIndices.push_back(zeroBasedIndex);
      }

      mlir::Value result = builder().create<SubscriptionOp>(loc, args[0], zeroBasedIndices);
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

    return lowerOperation<OperationKind::powerOf>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 2);
      mlir::Value base = args[0];
      mlir::Value exponent = args[1];

      mlir::Value result = builder().create<PowOp>(loc, resultType, base, exponent);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::powerOfEW(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::powerOfEW>(operation, [&](mlir::Location loc, mlir::ValueRange args) -> Results {
      assert(args.size() == 2);
      mlir::Value base = args[0];
      mlir::Value exponent = args[1];

      mlir::Value result = builder().create<PowOp>(loc, resultType, base, exponent);
      return Reference::ssa(&builder(), result);
    });
  }
}
