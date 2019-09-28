#pragma once

#include <cassert>
#include <memory>
#include <variant>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/lowerer/Constant.hpp"
#include "modelica/lowerer/Type.hpp"

namespace modelica
{
	enum class ExpressionKind
	{
		zero,
		negate,

		add,
		sub,
		mult,
		divide,
		greaterThan,
		greaterEqual,
		equal,
		different,
		less,
		lessEqual,
		elevation,
		module,

		conditional
	};

	class Expression
	{
		public:
		class Operation
		{
			public:
			Operation(
					ExpressionKind kind,
					std::unique_ptr<Expression> lhs,
					std::unique_ptr<Expression> rhs = nullptr,
					std::unique_ptr<Expression> cond = nullptr)
					: kind(kind),
						leftHandExpression(std::move(lhs)),
						rightHandExpression(std::move(rhs)),
						condition(std::move(cond))
			{
			}

			[[nodiscard]] bool isUnary() const
			{
				return kind >= ExpressionKind::negate && kind <= ExpressionKind::negate;
			}

			[[nodiscard]] bool isBinary() const
			{
				return kind >= ExpressionKind::add && kind <= ExpressionKind::module;
			}

			[[nodiscard]] bool isTernary() const
			{
				return kind >= ExpressionKind::conditional &&
							 kind <= ExpressionKind::conditional;
			}

			[[nodiscard]] const Expression& getLeftHand() const
			{
				assert(isUnary() || isBinary() || isTernary());	// NOLINT
				return *leftHandExpression;
			}

			[[nodiscard]] Expression& getLeftHand()
			{
				assert(isUnary() || isBinary() || isTernary());	// NOLINT
				return *leftHandExpression;
			}

			[[nodiscard]] const Expression& getRightHand() const
			{
				assert(isBinary() || isTernary());	// NOLINT
				return *rightHandExpression;
			}

			[[nodiscard]] Expression& getRightHand()
			{
				assert(isBinary() || isTernary());	// NOLINT
				return *rightHandExpression;
			}

			[[nodiscard]] Expression& getCondition()
			{
				assert(isTernary());	// NOLINT
				return *condition;
			}

			[[nodiscard]] const Expression& getCondition() const
			{
				assert(isTernary());	// NOLINT
				return *condition;
			}

			[[nodiscard]] ExpressionKind getKind() const { return kind; }

			Operation& operator=(Operation&& other)
			{
				kind = other.kind;
				leftHandExpression = std::move(other.leftHandExpression);
				rightHandExpression = std::move(other.rightHandExpression);
				condition = std::move(other.condition);
				return *this;
			}

			Operation(Operation&& other) = default;
			~Operation() = default;

			Operation(const Operation& other): kind(other.kind)
			{
				if (other.leftHandExpression != nullptr)
					leftHandExpression =
							std::make_unique<Expression>(*(other.leftHandExpression));
				if (other.rightHandExpression != nullptr)
					rightHandExpression =
							std::make_unique<Expression>(*(other.rightHandExpression));

				if (other.condition != nullptr)
					condition = std::make_unique<Expression>(*(other.condition));
			}

			Operation& operator=(const Operation& other)
			{
				kind = other.kind;
				if (other.leftHandExpression != nullptr)
					leftHandExpression =
							std::make_unique<Expression>(*(other.leftHandExpression));
				if (other.rightHandExpression != nullptr)
					rightHandExpression =
							std::make_unique<Expression>(*(other.rightHandExpression));

				if (other.condition != nullptr)
					condition = std::make_unique<Expression>(*(other.condition));
				return *this;
			}

			bool operator==(const Operation& other) const
			{
				if (other.kind != kind)
					return false;

				if (!deepEqual(
								leftHandExpression.get(), other.leftHandExpression.get()))
					return false;

				if (!deepEqual(
								rightHandExpression.get(), other.rightHandExpression.get()))
					return false;

				if (!deepEqual(condition.get(), other.condition.get()))
					return false;

				return true;
			}
			bool operator!=(const Operation& other) const
			{
				return !(*this == other);
			}

			private:
			bool deepEqual(Expression* first, Expression* second) const
			{
				if (first == nullptr && second != nullptr)
					return false;
				if (first != nullptr && second == nullptr)
					return false;
				if (first == nullptr && second == nullptr)
					return true;
				return *first == *second;
			}
			ExpressionKind kind{ ExpressionKind::zero };
			std::unique_ptr<Expression> leftHandExpression;
			std::unique_ptr<Expression> rightHandExpression;
			std::unique_ptr<Expression> condition;
		};

		Expression(std::string ref, Type returnType)
				: content(std::move(ref)), returnType(std::move(returnType))
		{
		}

		template<typename C>
		Expression(Constant<C> constant, Type returnType)
				: content(std::move(constant)), returnType(std::move(returnType))
		{
		}
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		[[nodiscard]] static Expression negate(Expression exp)
		{
			return Expression(
					ExpressionKind::negate, std::make_unique<Expression>(std::move(exp)));
		}

		[[nodiscard]] static Expression add(Expression lhs, Expression rhs)
		{
			return Expression(
					ExpressionKind::add,
					std::make_unique<Expression>(std::move(lhs)),
					std::make_unique<Expression>(std::move(rhs)));
		}

		[[nodiscard]] static Expression subtract(Expression lhs, Expression rhs)
		{
			return Expression(
					ExpressionKind::sub,
					std::make_unique<Expression>(std::move(lhs)),
					std::make_unique<Expression>(std::move(rhs)));
		}

		[[nodiscard]] static Expression multiply(Expression lhs, Expression rhs)
		{
			return Expression(
					ExpressionKind::mult,
					std::make_unique<Expression>(std::move(lhs)),
					std::make_unique<Expression>(std::move(rhs)));
		}

		[[nodiscard]] static Expression divide(Expression lhs, Expression rhs)
		{
			return Expression(
					ExpressionKind::divide,
					std::make_unique<Expression>(std::move(lhs)),
					std::make_unique<Expression>(std::move(rhs)));
		}

		[[nodiscard]] Expression operator!()
		{
			return Expression::negate(std::move(*this));
		}

		[[nodiscard]] Expression operator+(const Expression& other)
		{
			return Expression::add(std::move(*this), std::move(other));
		}

		[[nodiscard]] Expression operator-(const Expression& other)
		{
			return Expression::subtract(std::move(*this), std::move(other));
		}

		[[nodiscard]] Expression operator/(const Expression& other)
		{
			return Expression::divide(std::move(*this), std::move(other));
		}

		[[nodiscard]] bool isConstant() const
		{
			return !isOperation() && !isReference();
		}

		template<typename C>
		[[nodiscard]] bool isConstant() const
		{
			return std::holds_alternative<Constant<C>>(content);
		}

		[[nodiscard]] bool isOperation() const
		{
			return std::holds_alternative<Operation>(content);
		}

		template<typename C>
		[[nodiscard]] const Constant<C>& getConstant() const
		{
			assert(isConstant<C>());	// NOLINT
			return std::get<Constant<C>>(content);
		}

		template<typename C>
		[[nodiscard]] Constant<C>& getConstant()
		{
			assert(isConstant<C>());	// NOLINT
			return std::get<Constant<C>>(content);
		}

		[[nodiscard]] bool isUnary() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().isUnary();
		}

		[[nodiscard]] bool isBinary() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().isBinary();
		}

		[[nodiscard]] bool isTernary() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().isTernary();
		}

		[[nodiscard]] const Expression& getLeftHand() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getLeftHand();
		}

		[[nodiscard]] Expression& getLeftHand()
		{
			assert(isOperation());	// NOLINT
			return getOperation().getLeftHand();
		}

		[[nodiscard]] const Expression& getRightHand() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getRightHand();
		}

		[[nodiscard]] Expression& getRightHand()
		{
			assert(isOperation());	// NOLINT
			return getOperation().getRightHand();
		}

		[[nodiscard]] Expression& getCondition()
		{
			assert(isOperation());	// NOLINT
			return getOperation().getCondition();
		}

		[[nodiscard]] const Expression& getCondition() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getCondition();
		}

		[[nodiscard]] ExpressionKind getKind() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getKind();
		}

		bool operator==(const Expression& other) const
		{
			if (content != other.content)
				return false;
			return returnType == other.returnType;
		}
		bool operator!=(const Expression& other) const { return !(*this == other); }

		[[nodiscard]] const Type& getType() const { return returnType; }

		[[nodiscard]] bool isReference() const
		{
			return std::holds_alternative<std::string>(content);
		}

		[[nodiscard]] const std::string& getReference() const
		{
			assert(isReference());	// NOLINT
			return std::get<std::string>(content);
		}

		private:
		Expression(
				ExpressionKind kind,
				Type retType,
				std::unique_ptr<Expression> lhs,
				std::unique_ptr<Expression> rhs = nullptr,
				std::unique_ptr<Expression> cond = nullptr)
				: content(
							Operation(kind, std::move(lhs), std::move(rhs), std::move(cond))),
					returnType(std::move(retType))
		{
		}
		Expression(
				ExpressionKind kind,
				std::unique_ptr<Expression> lhs,
				std::unique_ptr<Expression> rhs = nullptr,
				std::unique_ptr<Expression> cond = nullptr)
				: content(
							Operation(kind, std::move(lhs), std::move(rhs), std::move(cond))),
					returnType(getLeftHand().getType())
		{
		}

		[[nodiscard]] const Operation& getOperation() const
		{
			assert(isOperation());	// NOLINT
			return std::get<Operation>(content);
		}

		[[nodiscard]] Operation& getOperation()
		{
			assert(isOperation());	// NOLINT
			return std::get<Operation>(content);
		}

		std::variant<
				Operation,
				IntConstant,
				BoolConstant,
				FloatConstant,
				std::string>
				content;
		Type returnType;
	};

	template<typename Expression, typename Visitor>
	void topDownVisit(Expression& exp, Visitor& visitor)
	{
		const auto visitChildren = [](Expression& exp, Visitor& visitor) {
			if (!exp.isOperation())
				return;
			topDownVisit(exp.getLeftHand(), visitor);

			if (exp.isBinary() || exp.isTernary())
				topDownVisit(exp.getRightHand(), visitor);

			if (exp.isTernary())
				topDownVisit(exp.getCondition(), visitor);
		};

		visitor.visit(exp);
		visitChildren(exp, visitor);
		visitor.afterVisit(exp);
	}

	template<typename Expression, typename Visitor>
	void bottomUpVisit(Expression& exp, Visitor& visitor)
	{
		const auto visitChildren = [](Expression& exp, Visitor& visitor) {
			if (!exp.isOperation())
				return;
			bottomUpVisit(exp.getLeftHand(), visitor);

			if (exp.isBinary() || exp.isTernary())
				bottomUpVisit(exp.getRightHand(), visitor);

			if (exp.isTernary())
				bottomUpVisit(exp.getCondition(), visitor);
		};

		visitChildren(exp, visitor);
		visitor.visit(exp);
	}

}	// namespace modelica
