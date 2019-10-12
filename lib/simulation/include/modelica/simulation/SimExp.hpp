#pragma once

#include <cassert>
#include <memory>
#include <variant>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/simulation/SimConst.hpp"
#include "modelica/simulation/SimType.hpp"

namespace modelica
{
	enum class SimExpKind
	{
		zero,

		negate,	 // unary expressions

		add,	// binary expressions
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

		conditional	 // thernary expressions
	};

	/**
	 * This is a compile time polimorfic class that can be any kind of expression
	 * costants, references, calls...
	 *
	 * There is no need to perform manual dynamic cast, all the informations
	 * can be queried with the getter such as isConstant, there are no derived.
	 *
	 * SimExp is standard type and can be copied, moved and compared, but it's
	 * large enough that it's not cheap to copy, and if copied instead of moved
	 * all the sub elements will be copied as well. So try to move and copy it
	 * as little as possible and pass it as reference.
	 *
	 */
	class SimExp
	{
		public:
		/**
		 * An operation is a class that holds all the informations regarding how to
		 * calculate the value of an expression that cointains subexpressions.
		 */
		class Operation
		{
			public:
			Operation(
					SimExpKind kind,
					std::unique_ptr<SimExp> lhs,
					std::unique_ptr<SimExp> rhs = nullptr,
					std::unique_ptr<SimExp> cond = nullptr)
					: kind(kind),
						leftHandExpression(std::move(lhs)),
						rightHandExpression(std::move(rhs)),
						condition(std::move(cond))
			{
			}
			/**
			 * \return 1 if it's a unary op, 2 if it's binary op
			 * 3 if it's ternary
			 */
			[[nodiscard]] size_t getArity() const { return arityOfOp(kind); }

			static size_t arityOfOp(SimExpKind kind)
			{
				if (kind >= SimExpKind::negate && kind <= SimExpKind::negate)
					return 1;
				if (kind >= SimExpKind::add && kind <= SimExpKind::module)
					return 2;
				if (kind >= SimExpKind::conditional && kind <= SimExpKind::conditional)
					return 3;

				assert(false && "Unreachable");	 // NOLINT
				return 0;
			}

			/**
			 * \return true if the operation only require one parameter.
			 */
			[[nodiscard]] bool isUnary() const { return arityOfOp(kind) == 1; }

			/**
			 * \return true if the operation require exactly two parameters.
			 */
			[[nodiscard]] bool isBinary() const { return arityOfOp(kind) == 2; }

			/**
			 * \return true if the operation requires exactly three parameters.
			 */
			[[nodiscard]] bool isTernary() const { return arityOfOp(kind) == 3; }

			/**
			 * \return the first sub expression, that is the only
			 * expression of a unary expression and the left argument
			 * of binary and thernary operations.
			 */
			[[nodiscard]] const SimExp& getLeftHand() const
			{
				assert(isUnary() || isBinary() || isTernary());	 // NOLINT
				return *leftHandExpression;
			}

			[[nodiscard]] SimExp& getLeftHand()
			{
				assert(isUnary() || isBinary() || isTernary());	 // NOLINT
				return *leftHandExpression;
			}

			/**
			 * \require isBinary() || isTernary()
			 *
			 * \return the second second element of the expression.
			 */
			[[nodiscard]] const SimExp& getRightHand() const
			{
				assert(isBinary() || isTernary());	// NOLINT
				return *rightHandExpression;
			}

			[[nodiscard]] SimExp& getRightHand()
			{
				assert(isBinary() || isTernary());	// NOLINT
				return *rightHandExpression;
			}

			/**
			 * \require isTernay()
			 *
			 * \return the conditional expression in a if esle expression
			 */
			[[nodiscard]] SimExp& getCondition()
			{
				assert(isTernary());	// NOLINT
				return *condition;
			}

			[[nodiscard]] const SimExp& getCondition() const
			{
				assert(isTernary());	// NOLINT
				return *condition;
			}

			[[nodiscard]] SimExpKind getKind() const { return kind; }

			/**
			 * \return the return type of the operation before being casted into the
			 * return type of the expression. As an example operation greaterThan may
			 * be casted into a float but the operation is resulting into a bool
			 */
			[[nodiscard]] SimType getOperationReturnType() const;

			Operation& operator=(Operation&& other) = default;

			Operation(Operation&& other) = default;
			~Operation() = default;

			/**
			 * Copy Constructor, copies the whole content of the other operation
			 */
			Operation(const Operation& other): kind(other.kind)
			{
				if (other.leftHandExpression != nullptr)
					leftHandExpression =
							std::make_unique<SimExp>(*(other.leftHandExpression));
				if (other.rightHandExpression != nullptr)
					rightHandExpression =
							std::make_unique<SimExp>(*(other.rightHandExpression));

				if (other.condition != nullptr)
					condition = std::make_unique<SimExp>(*(other.condition));
			}

			/**
			 * Move assigment operator
			 */
			Operation& operator=(const Operation& other)
			{
				if (this == &other)
					return *this;
				kind = other.kind;
				if (other.leftHandExpression != nullptr)
					leftHandExpression =
							std::make_unique<SimExp>(*(other.leftHandExpression));
				if (other.rightHandExpression != nullptr)
					rightHandExpression =
							std::make_unique<SimExp>(*(other.rightHandExpression));

				if (other.condition != nullptr)
					condition = std::make_unique<SimExp>(*(other.condition));
				return *this;
			}

			/**
			 * \return deep equality, check the two expressions are equivalent, that
			 * is every subelement is equal
			 */
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

			/**
			 * \return the inverse of operator ==
			 */
			bool operator!=(const Operation& other) const
			{
				return !(*this == other);
			}

			private:
			static bool deepEqual(SimExp* first, SimExp* second)
			{
				if (first == nullptr && second != nullptr)
					return false;
				if (first != nullptr && second == nullptr)
					return false;
				if (first == nullptr && second == nullptr)
					return true;
				return *first == *second;
			}
			SimExpKind kind{ SimExpKind::zero };
			std::unique_ptr<SimExp> leftHandExpression;
			std::unique_ptr<SimExp> rightHandExpression;
			std::unique_ptr<SimExp> condition;
		};

		/**
		 * Builds a reference expression based on the provided name
		 * of a var, and the type in which the values of that var will be casted.
		 */
		template<typename... T>
		SimExp(std::string ref, BultinSimTypes type, T... dimensions)
				: content(std::move(ref)), returnSimType(type, dimensions...)
		{
		}

		/**
		 * Builds a constant expression with the provided constant and the
		 * specified type in which the constant will be casted into. By default the
		 * type is deduced from the constant
		 */
		template<typename C>
		SimExp(
				SimConst<C> constant,
				SimType returnSimType = SimType(typeToBuiltin<C>()))
				: content(std::move(constant)), returnSimType(std::move(returnSimType))
		{
		}

		/**
		 * \return true if it's not an operation or if every subexpression is
		 * compatible
		 */
		[[nodiscard]] bool areSubExpressionCompatibles() const
		{
			if (!isOperation())
				return true;

			auto compatible =
					getLeftHand().getSimType().canBeCastedInto(getSimType());

			if (!isBinary())
				return compatible;
			return compatible &&
						 getRightHand().getSimType().canBeCastedInto(getSimType());
		}

		/**
		 * \brief Dumps the expression value into a human readable from to the
		 * provided raw_ostream by default it's standard out
		 */
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		/**
		 * \brief Builds a expression that is the negation of the provided one.
		 * \warning Remember to move the expression instead of copying them each
		 * time.
		 */
		[[nodiscard]] static SimExp negate(SimExp exp)
		{
			return SimExp(
					SimExpKind::negate, std::make_unique<SimExp>(std::move(exp)));
		}

		/**
		 * \brief Creates a add expression based on two values.
		 * \warning Remember to move the two expression instead of copying them each
		 * time.
		 */
		[[nodiscard]] static SimExp add(SimExp lhs, SimExp rhs)
		{
			return SimExp(
					SimExpKind::add,
					std::make_unique<SimExp>(std::move(lhs)),
					std::make_unique<SimExp>(std::move(rhs)));
		}

		/**
		 * \brief Creates a minus expression based on two values.
		 * \warning Remember to move the two expression instead of copying them each
		 * time.
		 */
		[[nodiscard]] static SimExp subtract(SimExp lhs, SimExp rhs)
		{
			return SimExp(
					SimExpKind::sub,
					std::make_unique<SimExp>(std::move(lhs)),
					std::make_unique<SimExp>(std::move(rhs)));
		}

		/**
		 * \brief Creates a multiply expression based on two values.
		 * \warning Remember to move the two expression instead of copying them each
		 * time.
		 */
		[[nodiscard]] static SimExp multiply(SimExp lhs, SimExp rhs)
		{
			return SimExp(
					SimExpKind::mult,
					std::make_unique<SimExp>(std::move(lhs)),
					std::make_unique<SimExp>(std::move(rhs)));
		}

		/**
		 * \brief Creates a divisions expression based on two values.
		 * \warning Remember to move the two expression instead of copying them each
		 * time.
		 */
		[[nodiscard]] static SimExp divide(SimExp lhs, SimExp rhs)
		{
			return SimExp(
					SimExpKind::divide,
					std::make_unique<SimExp>(std::move(lhs)),
					std::make_unique<SimExp>(std::move(rhs)));
		}

		/**
		 * \brief Creates a powerOf expression based on two values.
		 * \warning Remember to move the two expression instead of copying them each
		 * time.
		 */
		[[nodiscard]] static SimExp elevate(SimExp lhs, SimExp rhs)
		{
			return SimExp(
					SimExpKind::elevation,
					std::make_unique<SimExp>(std::move(lhs)),
					std::make_unique<SimExp>(std::move(rhs)));
		}

		/**
		 * \brief Creates a modulo expression based on two values.
		 * \warning Remember to move the two expression instead of copying them each
		 * time.
		 */
		[[nodiscard]] static SimExp modulo(SimExp lhs, SimExp rhs)
		{
			return SimExp(
					SimExpKind::module,
					std::make_unique<SimExp>(std::move(lhs)),
					std::make_unique<SimExp>(std::move(rhs)));
		}

		/**
		 * \brief Creates a modulo expression based on two values.
		 * \warning Remember to move the two expression instead of copying them each
		 * time.
		 */
		[[nodiscard]] static SimExp cond(SimExp cond, SimExp lhs, SimExp rhs)
		{
			return SimExp(
					SimExpKind::conditional,
					std::make_unique<SimExp>(std::move(lhs)),
					std::make_unique<SimExp>(std::move(rhs)),
					std::make_unique<SimExp>(std::move(cond)));
		}

		/**
		 * Creates a greather than expression, the return type has the same
		 * dimensions as left hand value but of bools
		 */
		[[nodiscard]] static SimExp greaterThan(SimExp lhs, SimExp rhs)
		{
			auto type = lhs.getSimType().as(BultinSimTypes::BOOL);
			return SimExp(
					SimExpKind::greaterThan,
					std::move(type),
					std::make_unique<SimExp>(std::move(lhs)),
					std::make_unique<SimExp>(std::move(rhs)));
		}

		/**
		 * Creates a greather equal expression, the return type has the same
		 * dimensions as left hand value but of bools
		 */
		[[nodiscard]] static SimExp greaterEqual(SimExp lhs, SimExp rhs)
		{
			auto type = lhs.getSimType().as(BultinSimTypes::BOOL);
			return SimExp(
					SimExpKind::greaterEqual,
					std::move(type),
					std::make_unique<SimExp>(std::move(lhs)),
					std::make_unique<SimExp>(std::move(rhs)));
		}

		/**
		 * Creates a less equal expression, the return type has the same
		 * dimensions as left hand value but of bools
		 */
		[[nodiscard]] static SimExp lessEqual(SimExp lhs, SimExp rhs)
		{
			auto type = lhs.getSimType().as(BultinSimTypes::BOOL);
			return SimExp(
					SimExpKind::lessEqual,
					std::move(type),
					std::make_unique<SimExp>(std::move(lhs)),
					std::make_unique<SimExp>(std::move(rhs)));
		}
		/**
		 * Creates a less expression, the return type has the same
		 * dimensions as left hand value but of bools
		 */
		[[nodiscard]] static SimExp lessThan(SimExp lhs, SimExp rhs)
		{
			auto type = lhs.getSimType().as(BultinSimTypes::BOOL);
			return SimExp(
					SimExpKind::less,
					std::move(type),
					std::make_unique<SimExp>(std::move(lhs)),
					std::make_unique<SimExp>(std::move(rhs)));
		}

		/**
		 * \brief Short hand for SimExp::greaterEqual
		 *
		 * \warning Notice that using the short hand will move the content no matter
		 * what
		 */
		[[nodiscard]] SimExp operator>=(const SimExp& other)
		{
			return SimExp::greaterThan(std::move(*this), std::move(other));
		}

		/**
		 * \brief Short hand for SimExp::greaterThan
		 *
		 * \warning Notice that using the short hand will move the content no matter
		 * what
		 */
		[[nodiscard]] SimExp operator>(const SimExp& other)
		{
			return SimExp::greaterThan(std::move(*this), std::move(other));
		}

		/**
		 * \brief Short hand for SimExp::lessThan
		 *
		 * \warning Notice that using the short hand will move the content no matter
		 * what
		 */
		[[nodiscard]] SimExp operator<(const SimExp& other)
		{
			return SimExp::lessThan(std::move(*this), std::move(other));
		}

		/**
		 * \brief Short hand for SimExp::lessEqual
		 *
		 * \warning Notice that using the short hand will move the content no matter
		 * what
		 */
		[[nodiscard]] SimExp operator<=(const SimExp& other)
		{
			return SimExp::lessEqual(std::move(*this), std::move(other));
		}

		/**
		 * \brief Short hand for SimExp::negate
		 *
		 * \warning Notice that using the short hand will move the content no matter
		 * what
		 */
		[[nodiscard]] SimExp operator!()
		{
			return SimExp::negate(std::move(*this));
		}

		/**
		 * \brief Short hand for SimExp::add
		 *
		 * \warning Notice that using the short hand will move the content no matter
		 * what.
		 */
		[[nodiscard]] SimExp operator+(const SimExp& other)
		{
			return SimExp::add(std::move(*this), std::move(other));
		}

		/**
		 * \brief Short hand for SimExp::subtract
		 *
		 * \warning Notice that using the short hand will move the content no matter
		 * what.
		 */
		[[nodiscard]] SimExp operator-(const SimExp& other)
		{
			return SimExp::subtract(std::move(*this), std::move(other));
		}

		/**
		 * \brief Short hand for SimExp::divide
		 *
		 * \warning Notice that using the short hand will move the content no matter
		 * what.
		 */
		[[nodiscard]] SimExp operator/(const SimExp& other)
		{
			return SimExp::divide(std::move(*this), std::move(other));
		}

		/**
		 * \brief Short hand for SimExp::multiply
		 *
		 * \warning Notice that using the short hand will move the content no matter
		 * what.
		 */
		[[nodiscard]] SimExp operator*(const SimExp& other)
		{
			return SimExp::multiply(std::move(*this), std::move(other));
		}

		/**
		 * \return true if the expression is a holding any kind of constant
		 */
		[[nodiscard]] bool isConstant() const
		{
			return !isOperation() && !isReference();
		}

		/**
		 * \return true if the expression is holding the expression of type C
		 */
		template<typename C>
		[[nodiscard]] bool isConstant() const
		{
			return std::holds_alternative<SimConst<C>>(content);
		}

		/**
		 * \return true if the expression is holding an operation
		 */
		[[nodiscard]] bool isOperation() const
		{
			return std::holds_alternative<Operation>(content);
		}

		/**
		 * \pre isConstant<C>()
		 * \return the constant holded by this expression.
		 */
		template<typename C>
		[[nodiscard]] const SimConst<C>& getConstant() const
		{
			assert(isConstant<C>());	// NOLINT
			return std::get<SimConst<C>>(content);
		}

		/**
		 * \pre isConstant<C>()
		 * \return the constant holded by this expression.
		 */
		template<typename C>
		[[nodiscard]] SimConst<C>& getConstant()
		{
			assert(isConstant<C>());	// NOLINT
			return std::get<SimConst<C>>(content);
		}

		/**
		 * \pre isOperation()
		 * \return true if the operation is unary
		 */
		[[nodiscard]] bool isUnary() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().isUnary();
		}

		/**
		 * \pre isOperation()
		 * \return true if the operation is binary
		 */
		[[nodiscard]] bool isBinary() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().isBinary();
		}

		/**
		 * \pre isOperation()
		 * \return true if the operation is ternary
		 */
		[[nodiscard]] bool isTernary() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().isTernary();
		}

		/**
		 * \pre isOperation()
		 *
		 * \return the type of the expression before being casted
		 * into the return type
		 */
		[[nodiscard]] SimType getOperationReturnType() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getOperationReturnType();
		}

		/**
		 * \pre isOperation()
		 * \return the left hand expression, or the only subexpression
		 * if it's a unary expression
		 */
		[[nodiscard]] const SimExp& getLeftHand() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getLeftHand();
		}

		/**
		 * \pre isOperation()
		 * \return the left hand expression, or the only subexpression
		 * if it's a unary expression
		 */
		[[nodiscard]] SimExp& getLeftHand()
		{
			assert(isOperation());	// NOLINT
			return getOperation().getLeftHand();
		}

		/**
		 * \pre isOperation()
		 * \return the right hand expression
		 */
		[[nodiscard]] const SimExp& getRightHand() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getRightHand();
		}

		/**
		 * \pre isOperation()
		 * \return the right hand expression
		 */
		[[nodiscard]] SimExp& getRightHand()
		{
			assert(isOperation());	// NOLINT
			return getOperation().getRightHand();
		}

		/**
		 * \pre isOperation()
		 * \return the condition of a conditional expression
		 */
		[[nodiscard]] SimExp& getCondition()
		{
			assert(isOperation());	// NOLINT
			return getOperation().getCondition();
		}

		/**
		 * \pre isOperation()
		 * \return the condition of a conditional expression
		 */
		[[nodiscard]] const SimExp& getCondition() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getCondition();
		}

		/**
		 * \pre isOperation()
		 * \return The kind of the expression
		 */
		[[nodiscard]] SimExpKind getKind() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getKind();
		}

		/**
		 * \return True iff the two expression are deeply equal, that is if every
		 * sub expression is equal to the other subexpressions.
		 */
		bool operator==(const SimExp& other) const
		{
			if (content != other.content)
				return false;
			return returnSimType == other.returnSimType;
		}

		/**
		 * \return The inverse of operator ==
		 */
		bool operator!=(const SimExp& other) const { return !(*this == other); }

		/**
		 * \return The type in which this expression will be casted into.
		 */
		[[nodiscard]] const SimType& getSimType() const { return returnSimType; }

		/**
		 * \return True if it's a reference to a variable
		 */
		[[nodiscard]] bool isReference() const
		{
			return std::holds_alternative<std::string>(content);
		}

		/**
		 * \return True if it's a reference to a variable
		 */
		[[nodiscard]] const std::string& getReference() const
		{
			assert(isReference());	// NOLINT
			return std::get<std::string>(content);
		}

		/**
		 * \return the arity of the operation
		 */
		[[nodiscard]] size_t getArity() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getArity();
		}

		private:
		SimExp(
				SimExpKind kind,
				SimType retSimType,
				std::unique_ptr<SimExp> lhs,
				std::unique_ptr<SimExp> rhs = nullptr,
				std::unique_ptr<SimExp> cond = nullptr)
				: content(
							Operation(kind, std::move(lhs), std::move(rhs), std::move(cond))),
					returnSimType(std::move(retSimType))
		{
			assert(!isOperation() || areSubExpressionCompatibles());	// NOLINT
		}
		SimExp(
				SimExpKind kind,
				std::unique_ptr<SimExp> lhs,
				std::unique_ptr<SimExp> rhs = nullptr,
				std::unique_ptr<SimExp> cond = nullptr)
				: content(
							Operation(kind, std::move(lhs), std::move(rhs), std::move(cond))),
					returnSimType(getLeftHand().getSimType())
		{
			assert(!isOperation() || areSubExpressionCompatibles());	// NOLINT
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
				IntSimConst,
				BoolSimConst,
				FloatSimConst,
				std::string>
				content;
		SimType returnSimType;
	};

	/**
	 * Visitor is class that must implement void visit(SimExp&) and
	 * void afterVisit(SimExp&). visit will be called for every sub expression of
	 * exp (exp included) if top down order. AfterVisit will be invoked after each
	 * children has been visited.
	 *
	 * You can implement a empty afterVisit to obtain a topDown visitor and not
	 * implement visit to get a bottomUpVisitor.
	 */
	template<typename SimExp, typename Visitor>
	void visit(SimExp& exp, Visitor& visitor)
	{
		const auto visitChildren = [](SimExp& exp, Visitor& visitor) {
			if (!exp.isOperation())
				return;
			visit(exp.getLeftHand(), visitor);

			if (exp.isBinary() || exp.isTernary())
				visit(exp.getRightHand(), visitor);

			if (exp.isTernary())
				visit(exp.getCondition(), visitor);
		};

		visitor.visit(exp);
		visitChildren(exp, visitor);
		visitor.afterVisit(exp);
	}

}	 // namespace modelica
