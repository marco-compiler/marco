#include "marco/AST/AST.h"
#include "marco/AST/Passes/ConstantFoldingPass.h"
#include <cmath>

using namespace ::marco;
using namespace ::marco::ast;

namespace
{
  template<BuiltInType Type>
  std::unique_ptr<Expression> foldNegateOp_scalar(SourceRange loc, const Expression& operand)
  {
    assert(operand.getType().isScalar());

    return Expression::constant(
        loc, makeType<Type>(),
        static_cast<frontendTypeToType_v<Type>>(-1 * operand.get<Constant>()->as<Type>()));
  }

  template<>
  std::unique_ptr<Expression> foldNegateOp_scalar<BuiltInType::Boolean>(SourceRange loc, const Expression& operand)
  {
    assert(operand.getType().isScalar());

    return Expression::constant(
        loc, makeType<BuiltInType::Boolean>(),
        !operand.get<Constant>()->as<BuiltInType::Boolean>());
  }

  template<BuiltInType Type>
  std::unique_ptr<Expression> foldAddOp_scalars(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<Type>(),
        static_cast<frontendTypeToType_v<Type>>(lhs.get<Constant>()->as<Type>() + rhs.get<Constant>()->as<Type>()));
  }

  template<>
  std::unique_ptr<Expression> foldAddOp_scalars<BuiltInType::Boolean>(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<BuiltInType::Boolean>(),
        lhs.get<Constant>()->as<BuiltInType::Boolean>() || rhs.get<Constant>()->as<BuiltInType::Boolean>());
  }

  template<BuiltInType Type>
  std::unique_ptr<Expression> foldSubOp_scalars(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<Type>(),
        static_cast<frontendTypeToType_v<Type>>(lhs.get<Constant>()->as<Type>() - rhs.get<Constant>()->as<Type>()));
  }

  template<>
  std::unique_ptr<Expression> foldSubOp_scalars<BuiltInType::Boolean>(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<BuiltInType::Boolean>(),
        lhs.get<Constant>()->as<BuiltInType::Boolean>() - rhs.get<Constant>()->as<BuiltInType::Boolean>());
  }

  template<BuiltInType Type>
  std::unique_ptr<Expression> foldMulOp_scalars(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<Type>(),
        static_cast<frontendTypeToType_v<Type>>(lhs.get<Constant>()->as<Type>() * rhs.get<Constant>()->as<Type>()));
  }

  template<>
  std::unique_ptr<Expression> foldMulOp_scalars<BuiltInType::Boolean>(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<BuiltInType::Boolean>(),
        lhs.get<Constant>()->as<BuiltInType::Boolean>() && rhs.get<Constant>()->as<BuiltInType::Boolean>());
  }

  template<BuiltInType Type>
  std::unique_ptr<Expression> foldDivOp_scalars(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<Type>(),
        static_cast<frontendTypeToType_v<Type>>(lhs.get<Constant>()->as<Type>() / rhs.get<Constant>()->as<Type>()));
  }

  template<BuiltInType Type>
  std::unique_ptr<Expression> foldPowOp_scalars(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<Type>(),
        static_cast<frontendTypeToType_v<Type>>(std::pow(lhs.get<Constant>()->as<Type>(), rhs.get<Constant>()->as<Type>())));
  }

  template<BuiltInType Type>
  std::unique_ptr<Expression> foldEqualOp(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<Type>(),
        static_cast<frontendTypeToType_v<Type>>(lhs.get<Constant>()->as<Type>() == rhs.get<Constant>()->as<Type>()));
  }

  template<BuiltInType Type>
  std::unique_ptr<Expression> foldNotEqualOp(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<Type>(),
        static_cast<frontendTypeToType_v<Type>>(lhs.get<Constant>()->as<Type>() != rhs.get<Constant>()->as<Type>()));
  }

  template<BuiltInType Type>
  std::unique_ptr<Expression> foldGreaterOp(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<Type>(),
        static_cast<frontendTypeToType_v<Type>>(lhs.get<Constant>()->as<Type>() > rhs.get<Constant>()->as<Type>()));
  }

  template<BuiltInType Type>
  std::unique_ptr<Expression> foldGreaterEqualOp(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<Type>(),
        static_cast<frontendTypeToType_v<Type>>(lhs.get<Constant>()->as<Type>() >= rhs.get<Constant>()->as<Type>()));
  }

  template<BuiltInType Type>
  std::unique_ptr<Expression> foldLessOp(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<Type>(),
        static_cast<frontendTypeToType_v<Type>>(lhs.get<Constant>()->as<Type>() < rhs.get<Constant>()->as<Type>()));
  }

  template<BuiltInType Type>
  std::unique_ptr<Expression> foldLessEqualOp(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<Type>(),
        static_cast<frontendTypeToType_v<Type>>(lhs.get<Constant>()->as<Type>() <= rhs.get<Constant>()->as<Type>()));
  }

  template<BuiltInType Type>
  std::unique_ptr<Expression> foldAndOp(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<Type>(),
        static_cast<frontendTypeToType_v<Type>>(lhs.get<Constant>()->as<Type>() && rhs.get<Constant>()->as<Type>()));
  }

  template<BuiltInType Type>
  std::unique_ptr<Expression> foldOrOp(SourceRange loc, const Expression& lhs, const Expression& rhs)
  {
    assert(lhs.getType().isScalar());
    assert(rhs.getType().isScalar());

    return Expression::constant(
        loc, makeType<Type>(),
        static_cast<frontendTypeToType_v<Type>>(lhs.get<Constant>()->as<Type>() || rhs.get<Constant>()->as<Type>()));
  }
}

namespace marco::ast
{
  ConstantFoldingPass::ConstantFoldingPass(diagnostic::DiagnosticEngine& diagnostics)
      : Pass(diagnostics)
  {
  }

  template<>
  bool ConstantFoldingPass::run<Class>(Class& cls)
  {
    return cls.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(cls);
    });
  }

  bool ConstantFoldingPass::run(std::unique_ptr<Class>& cls)
  {
    return run<Class>(*cls);
  }

  template<>
  bool ConstantFoldingPass::run<PartialDerFunction>(Class& cls)
  {
    return true;
  }

  template<>
  bool ConstantFoldingPass::run<StandardFunction>(Class& cls)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* function = cls.get<StandardFunction>();

    for (auto& member : function->getMembers()) {
      if (!run(*member)) {
        return false;
      }
    }

    for (auto& algorithm : function->getAlgorithms()) {
      if (!run(*algorithm)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::run<Model>(Class& cls)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* model = cls.get<Model>();

    for (auto& innerClass : model->getInnerClasses()) {
      if (!run<Class>(*innerClass)) {
        return false;
      }
    }

    for (auto& member : model->getMembers()) {
      if (!run(*member)) {
        return false;
      }
    }

    for (auto& equationsBlock : model->getEquationsBlocks()) {
      for (auto& equation : equationsBlock->getEquations()) {
        if (!run(*equation)) {
          return false;
        }
      }

      for (auto& forEquation : equationsBlock->getForEquations()) {
        if (!run(*forEquation)) {
          return false;
        }
      }
    }

    for (auto& equationsBlock : model->getInitialEquationsBlocks()) {
      for (auto& equation : equationsBlock->getEquations()) {
        if (!run(*equation)) {
          return false;
        }
      }

      for (auto& forEquation : equationsBlock->getForEquations()) {
        if (!run(*forEquation)) {
          return false;
        }
      }
    }

    for (auto& algorithm : model->getAlgorithms()) {
      if (!run(*algorithm)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::run<Package>(Class& cls)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* package = cls.get<Package>();

    for (auto& innerClass : *package) {
      if (!run<Class>(*innerClass)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::run<Record>(Class& cls)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* record = cls.get<Record>();

    for (auto& member : *record) {
      if (!run(*member)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::run<Expression>(Expression& expression)
  {
    return expression.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(expression);
    });
  }

  template<>
  bool ConstantFoldingPass::run<Array>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* array = expression.get<Array>();

    for (auto& element : *array) {
      if (!run<Expression>(*element)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::run<Call>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* call = expression.get<Call>();

    for (auto& arg : *call) {
      if (!run<Expression>(*arg)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::run<Constant>(Expression& expression)
  {
    return true;
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::negate>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 1);

    auto* operand = operation->getArg(0);

    if (operand->isa<Constant>()) {
      assert(operand->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (type == BuiltInType::Boolean) {
        expression = std::move(*foldNegateOp_scalar<BuiltInType::Boolean>(expression.getLocation(), *operand));
      } else if (type == BuiltInType::Integer) {
        expression = std::move(*foldNegateOp_scalar<BuiltInType::Integer>(expression.getLocation(), *operand));
      } else if (type == BuiltInType::Real) {
        expression = std::move(*foldNegateOp_scalar<BuiltInType::Real>(expression.getLocation(), *operand));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::add>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto* lhs = operation->getArg(0);
    auto* rhs = operation->getArg(1);

    if (lhs->isa<Constant>() && rhs->isa<Constant>()) {
      assert(lhs->getType().isa<BuiltInType>() && rhs->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (type == BuiltInType::Boolean) {
        expression = std::move(*foldAddOp_scalars<BuiltInType::Boolean>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Integer) {
        expression = std::move(*foldAddOp_scalars<BuiltInType::Integer>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Real) {
        expression = std::move(*foldAddOp_scalars<BuiltInType::Real>(expression.getLocation(), *lhs, *rhs));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::subtract>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto* lhs = operation->getArg(0);
    auto* rhs = operation->getArg(1);

    if (lhs->isa<Constant>() && rhs->isa<Constant>()) {
      assert(lhs->getType().isa<BuiltInType>() && rhs->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (type == BuiltInType::Boolean) {
        expression = std::move(*foldSubOp_scalars<BuiltInType::Boolean>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Integer) {
        expression = std::move(*foldSubOp_scalars<BuiltInType::Integer>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Real) {
        expression = std::move(*foldSubOp_scalars<BuiltInType::Real>(expression.getLocation(), *lhs, *rhs));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::multiply>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto* lhs = operation->getArg(0);
    auto* rhs = operation->getArg(1);

    if (lhs->isa<Constant>() && rhs->isa<Constant>()) {
      assert(lhs->getType().isa<BuiltInType>() && rhs->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (type == BuiltInType::Boolean) {
        expression = std::move(*foldMulOp_scalars<BuiltInType::Boolean>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Integer) {
        expression = std::move(*foldMulOp_scalars<BuiltInType::Integer>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Real) {
        expression = std::move(*foldMulOp_scalars<BuiltInType::Real>(expression.getLocation(), *lhs, *rhs));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::divide>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto* lhs = operation->getArg(0);
    auto* rhs = operation->getArg(1);

    if (lhs->isa<Constant>() && rhs->isa<Constant>()) {
      assert(lhs->getType().isa<BuiltInType>() && rhs->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (type == BuiltInType::Boolean) {
        expression = std::move(*foldDivOp_scalars<BuiltInType::Boolean>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Integer) {
        expression = std::move(*foldDivOp_scalars<BuiltInType::Integer>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Real) {
        expression = std::move(*foldDivOp_scalars<BuiltInType::Real>(expression.getLocation(), *lhs, *rhs));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::powerOf>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto* base = operation->getArg(0);
    auto* exponent = operation->getArg(1);

    if (base->isa<Constant>() && exponent->isa<Constant>()) {
      assert(base->getType().isa<BuiltInType>() && exponent->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (type == BuiltInType::Boolean) {
        expression = std::move(*foldPowOp_scalars<BuiltInType::Boolean>(expression.getLocation(), *base, *exponent));
      } else if (type == BuiltInType::Integer) {
        expression = std::move(*foldPowOp_scalars<BuiltInType::Integer>(expression.getLocation(), *base, *exponent));
      } else if (type == BuiltInType::Real) {
        expression = std::move(*foldPowOp_scalars<BuiltInType::Real>(expression.getLocation(), *base, *exponent));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::equal>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto* lhs = operation->getArg(0);
    auto* rhs = operation->getArg(1);

    if (lhs->isa<Constant>() && rhs->isa<Constant>()) {
      assert(lhs->getType().isa<BuiltInType>() && rhs->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (type == BuiltInType::Boolean) {
        expression = std::move(*foldEqualOp<BuiltInType::Boolean>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Integer) {
        expression = std::move(*foldEqualOp<BuiltInType::Integer>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Real) {
        expression = std::move(*foldEqualOp<BuiltInType::Real>(expression.getLocation(), *lhs, *rhs));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::different>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto* lhs = operation->getArg(0);
    auto* rhs = operation->getArg(1);

    if (lhs->isa<Constant>() && rhs->isa<Constant>()) {
      assert(lhs->getType().isa<BuiltInType>() && rhs->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (type == BuiltInType::Boolean) {
        expression = std::move(*foldNotEqualOp<BuiltInType::Boolean>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Integer) {
        expression = std::move(*foldNotEqualOp<BuiltInType::Integer>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Real) {
        expression = std::move(*foldNotEqualOp<BuiltInType::Real>(expression.getLocation(), *lhs, *rhs));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::greater>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto* lhs = operation->getArg(0);
    auto* rhs = operation->getArg(1);

    if (lhs->isa<Constant>() && rhs->isa<Constant>()) {
      assert(lhs->getType().isa<BuiltInType>() && rhs->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (type == BuiltInType::Boolean) {
        expression = std::move(*foldGreaterOp<BuiltInType::Boolean>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Integer) {
        expression = std::move(*foldGreaterOp<BuiltInType::Integer>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Real) {
        expression = std::move(*foldGreaterOp<BuiltInType::Real>(expression.getLocation(), *lhs, *rhs));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::greaterEqual>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto* lhs = operation->getArg(0);
    auto* rhs = operation->getArg(1);

    if (lhs->isa<Constant>() && rhs->isa<Constant>()) {
      assert(lhs->getType().isa<BuiltInType>() && rhs->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (type == BuiltInType::Boolean) {
        expression = std::move(*foldGreaterEqualOp<BuiltInType::Boolean>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Integer) {
        expression = std::move(*foldGreaterEqualOp<BuiltInType::Integer>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Real) {
        expression = std::move(*foldGreaterEqualOp<BuiltInType::Real>(expression.getLocation(), *lhs, *rhs));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::less>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto* lhs = operation->getArg(0);
    auto* rhs = operation->getArg(1);

    if (lhs->isa<Constant>() && rhs->isa<Constant>()) {
      assert(lhs->getType().isa<BuiltInType>() && rhs->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (type == BuiltInType::Boolean) {
        expression = std::move(*foldLessOp<BuiltInType::Boolean>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Integer) {
        expression = std::move(*foldLessOp<BuiltInType::Integer>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Real) {
        expression = std::move(*foldLessOp<BuiltInType::Real>(expression.getLocation(), *lhs, *rhs));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::lessEqual>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto* lhs = operation->getArg(0);
    auto* rhs = operation->getArg(1);

    if (lhs->isa<Constant>() && rhs->isa<Constant>()) {
      assert(lhs->getType().isa<BuiltInType>() && rhs->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (type == BuiltInType::Boolean) {
        expression = std::move(*foldLessEqualOp<BuiltInType::Boolean>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Integer) {
        expression = std::move(*foldLessEqualOp<BuiltInType::Integer>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Real) {
        expression = std::move(*foldLessEqualOp<BuiltInType::Real>(expression.getLocation(), *lhs, *rhs));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<BuiltInType Type>
  static std::unique_ptr<Expression> foldNotOp(
      SourceRange loc, const Expression& operand)
  {
    assert(operand.getType().isScalar());

    return Expression::constant(
        loc, makeType<Type>(),
        static_cast<frontendTypeToType_v<Type>>(operand.get<Constant>()->as<Type>() <= 0));
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::lnot>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 1);

    auto* operand = operation->getArg(0);

    if (operand->isa<Constant>()) {
      assert(operand->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (type == BuiltInType::Boolean) {
        expression = std::move(*foldNotOp<BuiltInType::Boolean>(expression.getLocation(), *operand));
      } else if (type == BuiltInType::Integer) {
        expression = std::move(*foldNotOp<BuiltInType::Integer>(expression.getLocation(), *operand));
      } else if (type == BuiltInType::Real) {
        expression = std::move(*foldNotOp<BuiltInType::Real>(expression.getLocation(), *operand));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::land>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto* lhs = operation->getArg(0);
    auto* rhs = operation->getArg(1);

    if (lhs->isa<Constant>() && rhs->isa<Constant>()) {
      assert(lhs->getType().isa<BuiltInType>() && rhs->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (type == BuiltInType::Boolean) {
        expression = std::move(*foldAndOp<BuiltInType::Boolean>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Integer) {
        expression = std::move(*foldAndOp<BuiltInType::Integer>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Real) {
        expression = std::move(*foldAndOp<BuiltInType::Real>(expression.getLocation(), *lhs, *rhs));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::lor>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 2);

    auto* lhs = operation->getArg(0);
    auto* rhs = operation->getArg(1);

    if (lhs->isa<Constant>() && rhs->isa<Constant>()) {
      assert(lhs->getType().isa<BuiltInType>() && rhs->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (type == BuiltInType::Boolean) {
        expression = std::move(*foldOrOp<BuiltInType::Boolean>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Integer) {
        expression = std::move(*foldOrOp<BuiltInType::Integer>(expression.getLocation(), *lhs, *rhs));
      } else if (type == BuiltInType::Real) {
        expression = std::move(*foldOrOp<BuiltInType::Real>(expression.getLocation(), *lhs, *rhs));
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::processOp<OperationKind::ifelse>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto* operation = expression.get<Operation>();
    assert(operation->argumentsCount() == 3);

    auto* condition = operation->getArg(0);
    auto* trueValue = operation->getArg(1);
    auto* falseValue = operation->getArg(2);

    if (condition->isa<Constant>()) {
      assert(condition->getType().isa<BuiltInType>());
      auto type = expression.getType();
      assert(type.isa<BuiltInType>());
      assert(type.get<BuiltInType>() != BuiltInType::Unknown);

      if (condition->get<Constant>()->as<BuiltInType::Boolean>()) {
        expression = std::move(*trueValue);
      } else {
        expression = std::move(*falseValue);
      }

      expression.setType(type);
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::run<Operation>(Expression& expression)
  {
    auto* operation = expression.get<Operation>();

    for (size_t i = 0; i < operation->argumentsCount(); ++i) {
      if (!run<Expression>(*operation->getArg(i))) {
        return false;
      }
    }

    switch (operation->getOperationKind()) {
      case OperationKind::add:
        return processOp<OperationKind::add>(expression);

      case OperationKind::addEW:
        return true;

      case OperationKind::different:
        return processOp<OperationKind::different>(expression);

      case OperationKind::divide:
        return processOp<OperationKind::divide>(expression);

      case OperationKind::divideEW:
        return true;

      case OperationKind::equal:
        return processOp<OperationKind::equal>(expression);

      case OperationKind::greater:
        return processOp<OperationKind::greater>(expression);

      case OperationKind::greaterEqual:
        return processOp<OperationKind::greaterEqual>(expression);

      case OperationKind::ifelse:
        return processOp<OperationKind::ifelse>(expression);

      case OperationKind::less:
        return processOp<OperationKind::less>(expression);

      case OperationKind::lessEqual:
        return processOp<OperationKind::lessEqual>(expression);

      case OperationKind::land:
        return processOp<OperationKind::land>(expression);

      case OperationKind::lnot:
        return processOp<OperationKind::lnot>(expression);

      case OperationKind::lor:
        return processOp<OperationKind::lor>(expression);

      case OperationKind::memberLookup:
        return true;

      case OperationKind::multiply:
        return processOp<OperationKind::multiply>(expression);

      case OperationKind::multiplyEW:
        return true;

      case OperationKind::negate:
        return processOp<OperationKind::negate>(expression);

      case OperationKind::powerOf:
        return processOp<OperationKind::powerOf>(expression);

      case OperationKind::powerOfEW:
        return true;

      case OperationKind::range:
        return true;

      case OperationKind::subscription:
        return true;

      case OperationKind::subtract:
        return processOp<OperationKind::subtract>(expression);

      case OperationKind::subtractEW:
        return true;
    }

    llvm_unreachable("Unknown operation kind");
    return false;
  }

  template<>
  bool ConstantFoldingPass::run<ReferenceAccess>(Expression& expression)
  {
    return true;
  }

  template<>
  bool ConstantFoldingPass::run<Tuple>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* tuple = expression.get<Tuple>();

    for (auto& exp : *tuple) {
      if (!run<Expression>(*exp)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::run<RecordInstance>(Expression& expression)
  {
    return true;
  }

  bool ConstantFoldingPass::run(Equation& equation)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    if (!run<Expression>(*equation.getLhsExpression())) {
      return false;
    }

    if (!run<Expression>(*equation.getRhsExpression())) {
      return false;
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  bool ConstantFoldingPass::run(ForEquation& forEquation)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    for (auto& induction : forEquation.getInductions()) {
      if (!run<Expression>(*induction->getBegin())) {
        return false;
      }

      if (!run<Expression>(*induction->getEnd())) {
        return false;
      }
    }

    if (!run(*forEquation.getEquation())) {
      return false;
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  bool ConstantFoldingPass::run(Induction& induction)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    if (!run<Expression>(*induction.getBegin())) {
      return false;
    }

    if (!run<Expression>(*induction.getBegin())) {
      return true;
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  bool ConstantFoldingPass::run(Member& member)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    auto& type = member.getType();

    for (auto& dimension : type.getDimensions()) {
      if (dimension.hasExpression()) {
        if (!run<Expression>(*dimension.getExpression())) {
          return false;
        }
      }
    }

    if (member.hasModification()) {
      if (!run(*member.getModification())) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::run<Statement>(Statement& statement)
  {
    return statement.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(statement);
    });
  }

  bool ConstantFoldingPass::run(Algorithm& algorithm)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    for (auto& statement : algorithm.getBody()) {
      if (!run<Statement>(*statement)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::run<AssignmentStatement>(Statement& statement)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* assignmentStatement = statement.get<AssignmentStatement>();

    if (!run<Expression>(*assignmentStatement->getDestinations())) {
      return false;
    }

    if (!run<Expression>(*assignmentStatement->getExpression())) {
      return false;
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::run<BreakStatement>(Statement& statement)
  {
    return true;
  }

  template<>
  bool ConstantFoldingPass::run<ForStatement>(Statement& statement)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* forStatement = statement.get<ForStatement>();

    if (!run(*forStatement->getInduction())) {
      return false;
    }

    for (auto& stmnt : forStatement->getBody()) {
      if (!run<Statement>(*stmnt)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::run<IfStatement>(Statement& statement)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* ifStatement = statement.get<IfStatement>();

    for (auto& block : *ifStatement) {
      if (!run<Expression>(*block.getCondition())) {
        return false;
      }

      for (auto& stmnt : block) {
        if (!run<Statement>(*stmnt)) {
          return false;
        }
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::run<ReturnStatement>(Statement& statement)
  {
    return true;
  }

  template<>
  bool ConstantFoldingPass::run<WhenStatement>(Statement& statement)
  {
    llvm_unreachable("Not implemented");
    return false;
  }

  template<>
  bool ConstantFoldingPass::run<WhileStatement>(Statement& statement)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* whileStatement = statement.get<WhileStatement>();

    if (!run<Expression>(*whileStatement->getCondition())) {
      return false;
    }

    for (auto& stmnt : whileStatement->getBody()) {
      if (!run<Statement>(*stmnt)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  bool ConstantFoldingPass::run(Modification& modification)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    if (modification.hasClassModification()) {
      if (!run(*modification.getClassModification())) {
        return false;
      }
    }

    if (modification.hasExpression()) {
      if (!run<Expression>(*modification.getExpression())) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::run<ElementModification>(Argument& argument)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* elementModification = argument.get<ElementModification>();

    if (elementModification->hasModification()) {
      if (!run(*elementModification->getModification())) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool ConstantFoldingPass::run<ElementRedeclaration>(Argument& argument)
  {
    llvm_unreachable("Not implemented");
    return false;
  }

  template<>
  bool ConstantFoldingPass::run<ElementReplaceable>(Argument& argument)
  {
    llvm_unreachable("Not implemented");
    return false;
  }

  template<>
  bool ConstantFoldingPass::run<Argument>(Argument& argument)
  {
    return argument.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(argument);
    });
  }

  bool ConstantFoldingPass::run(ClassModification& classModification)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    for (auto& argument : classModification) {
      if (!run<Argument>(*argument)) {
        return false;
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  std::unique_ptr<Pass> createConstantFoldingPass(diagnostic::DiagnosticEngine& diagnostics)
  {
    return std::make_unique<ConstantFoldingPass>(diagnostics);
  }
}
