#include "marco/Codegen/Lowering/OperationLowerer.h"

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

  std::vector<Results> OperationLowerer::lowerArgs(const Operation& operation)
  {
    std::vector<Results> args;

    for (const auto& arg : operation) {
      args.push_back(lower(*arg));
    }

    return args;
  }

  Results OperationLowerer::negate(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::negate>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 1);

      const auto& value = args[0];
      assert(value.size() == 1);

      mlir::Value result = builder().create<NegateOp>(loc, resultType, *value[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::add(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::add>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() >= 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<AddOp>(loc, resultType, *first[0], *second[0]);

      for (size_t i = 2; i < args.size(); ++i) {
        const auto& current = args[i];
        assert(current.size() == 1);
        result = builder().create<AddOp>(loc, resultType, result, *current[0]);
      }

      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::addEW(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::addEW>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<AddEWOp>(loc, resultType, *first[0], *second[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::subtract(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::subtract>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<SubOp>(loc, resultType, *first[0], *second[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::subtractEW(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::subtractEW>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<SubEWOp>(loc, resultType, *first[0], *second[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::multiply(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::multiply>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() >= 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<MulOp>(loc, resultType, *first[0], *second[0]);

      for (size_t i = 2; i < args.size(); ++i) {
        const auto& current = args[i];
        assert(current.size() == 1);

        result = builder().create<MulOp>(loc, resultType, result, *current[0]);
      }

      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::multiplyEW(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::multiplyEW>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<MulEWOp>(loc, resultType, *first[0], *second[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::divide(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::divide>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<DivOp>(loc, resultType, *first[0], *second[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::divideEW(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::divideEW>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<DivEWOp>(loc, resultType, *first[0], *second[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::ifElse(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::ifelse>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 3);

      const auto& condition = args[0];
      assert(condition.size() == 1);

      std::vector<mlir::Value> trueValues;

      for (const auto& trueValue : args[1]) {
        trueValues.push_back(*trueValue);
      }

      std::vector<mlir::Value> falseValues;

      for (const auto& falseValue : args[2]) {
        falseValues.push_back(*falseValue);
      }

      auto selectOp = builder().create<SelectOp>(loc, resultType, *condition[0], trueValues, falseValues);
      std::vector<Reference> results;

      for (const auto& result : selectOp.getResults()) {
        results.push_back(Reference::ssa(&builder(), result));
      }

      return Results(results.begin(), results.end());
    });
  }

  Results OperationLowerer::greater(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::greater>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<GtOp>(loc, resultType, *first[0], *second[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::greaterOrEqual(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::greaterEqual>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<GteOp>(loc, resultType, *first[0], *second[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::equal(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::equal>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<EqOp>(loc, resultType, *first[0], *second[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::notEqual(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::different>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<NotEqOp>(loc, resultType, *first[0], *second[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::lessOrEqual(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::lessEqual>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<LteOp>(loc, resultType, *first[0], *second[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::less(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::less>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<LtOp>(loc, resultType, *first[0], *second[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::logicalAnd(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::land>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<AndOp>(loc, resultType, *first[0], *second[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::logicalNot(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::lnot>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 1);

      const auto& value = args[0];
      assert(value.size() == 1);

      mlir::Value result = builder().create<NotOp>(loc, resultType, *value[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::logicalOr(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::lor>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& first = args[0];
      assert(first.size() == 1);

      const auto& second = args[1];
      assert(second.size() == 1);

      mlir::Value result = builder().create<OrOp>(loc, resultType, *first[0], *second[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::subscription(const ast::Operation& operation)
  {
    return lowerOperation<OperationKind::subscription>(
        operation,
        [&](mlir::Location loc, std::vector<Results> args) -> Results {
          assert(args.size() >= 1);

          const auto& array = args[0];
          assert(array.size() == 1);

          mlir::Value arrayValue = *array[0];
          assert(arrayValue.getType().isa<ArrayType>());

          // Indices in Modelica are 1-based, while in the MLIR dialect are
          // 0-based. Thus, we need to shift them by one. In doing so, we also
          // force the result to be of index type.
          std::vector<mlir::Value> zeroBasedIndices;

          for (size_t i = 1; i < args.size(); ++i) {
            const auto& index = args[i];
            assert(index.size() == 1);

            mlir::Value indexValue = *index[0];

            mlir::Value one = builder().create<ConstantOp>(
                indexValue.getLoc(), builder().getIndexAttr(-1));

            mlir::Value zeroBasedIndex = builder().create<AddOp>(
                indexValue.getLoc(),
                builder().getIndexType(),
                indexValue, one);

            zeroBasedIndices.push_back(zeroBasedIndex);
          }

          mlir::Value result = builder().create<SubscriptionOp>(
              loc, arrayValue, zeroBasedIndices);

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

    return lowerOperation<OperationKind::powerOf>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& base = args[0];
      assert(base.size() == 1);

      const auto& exponent = args[1];
      assert(exponent.size() == 1);

      mlir::Value result = builder().create<PowOp>(loc, resultType, *base[0], *exponent[0]);
      return Reference::ssa(&builder(), result);
    });
  }

  Results OperationLowerer::powerOfEW(const ast::Operation& operation)
  {
    mlir::Type resultType = lower(operation.getType());

    return lowerOperation<OperationKind::powerOfEW>(operation, [&](mlir::Location loc, std::vector<Results> args) -> Results {
      assert(args.size() == 2);

      const auto& base = args[0];
      assert(base.size() == 1);

      const auto& exponent = args[1];
      assert(exponent.size() == 1);

      mlir::Value result = builder().create<PowOp>(loc, resultType, *base[0], *exponent[0]);
      return Reference::ssa(&builder(), result);
    });
  }
}
