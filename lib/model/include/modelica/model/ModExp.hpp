#pragma once

#include <cassert>
#include <memory>
#include <variant>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/model/ModCall.hpp"
#include "modelica/model/ModChildrenIterator.hpp"
#include "modelica/model/ModConst.hpp"
#include "modelica/model/ModType.hpp"

namespace modelica
{
	enum class ModExpKind
	{
		zero,

		negate,	 // unary expressions
		induction,

		add,	// binary expressions
		sub,
		at,
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
	 * ModExp is standard type and can be copied, moved and compared, but it's
	 * large enough that it's not cheap to copy, and if copied instead of moved
	 * all the sub elements will be copied as well. So try to move and copy it
	 * as little as possible and pass it as reference.
	 *
	 */
	class ModExp
	{
		public:
		friend class ModParser;
		/**
		 * An operation is a class that holds all the informations regarding how to
		 * calculate the value of an expression that cointains subexpressions.
		 */
		class Operation
		{
			public:
			Operation(
					ModExpKind kind,
					std::unique_ptr<ModExp> lhs,
					std::unique_ptr<ModExp> rhs = nullptr,
					std::unique_ptr<ModExp> cond = nullptr)
					: kind(kind),
						leftHandExpression(std::move(lhs)),
						rightHandExpression(std::move(rhs)),
						condition(std::move(cond))
			{
				assert(leftHandExpression != nullptr);
				assert(!(getArity() == 2) || rightHandExpression != nullptr);
				assert(!(getArity() == 3) || condition != nullptr);
			}

			Operation(
					ModExpKind kind, llvm::SmallVector<std::unique_ptr<ModExp>, 3> exps)
					: kind(kind)
			{
				assert(exps.size() == arityOfOp(kind));	 // NOLINT

				if (arityOfOp(kind) > 0)
					leftHandExpression = move(exps[0]);
				if (arityOfOp(kind) > 1)
					rightHandExpression = move(exps[1]);
				if (arityOfOp(kind) > 2)
					condition = move(exps[2]);
			}
			/**
			 * \return 1 if it's a unary op, 2 if it's binary op
			 * 3 if it's ternary
			 */
			[[nodiscard]] size_t getArity() const { return arityOfOp(kind); }

			static size_t arityOfOp(ModExpKind kind)
			{
				if (kind >= ModExpKind::negate && kind <= ModExpKind::induction)
					return 1;
				if (kind >= ModExpKind::add && kind <= ModExpKind::module)
					return 2;
				if (kind >= ModExpKind::conditional && kind <= ModExpKind::conditional)
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
			[[nodiscard]] const ModExp& getLeftHand() const
			{
				assert(isUnary() || isBinary() || isTernary());	 // NOLINT
				return *leftHandExpression;
			}

			[[nodiscard]] ModExp& getLeftHand()
			{
				assert(isUnary() || isBinary() || isTernary());	 // NOLINT
				return *leftHandExpression;
			}

			/**
			 * \require isBinary() || isTernary()
			 *
			 * \return the second second element of the expression.
			 */
			[[nodiscard]] const ModExp& getRightHand() const
			{
				assert(isBinary() || isTernary());	// NOLINT
				return *rightHandExpression;
			}

			[[nodiscard]] ModExp& getRightHand()
			{
				assert(isBinary() || isTernary());	// NOLINT
				return *rightHandExpression;
			}

			/**
			 * \require isTernay()
			 *
			 * \return the conditional expression in a if esle expression
			 */
			[[nodiscard]] ModExp& getCondition()
			{
				assert(isTernary());	// NOLINT
				return *condition;
			}

			[[nodiscard]] const ModExp& getCondition() const
			{
				assert(isTernary());	// NOLINT
				return *condition;
			}

			[[nodiscard]] ModExpKind getKind() const { return kind; }

			/**
			 * \return the return type of the operation before being casted into the
			 * return type of the expression. As an example operation greaterThan may
			 * be casted into a float but the operation is resulting into a bool
			 */
			[[nodiscard]] ModType getOperationReturnType() const;

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
							std::make_unique<ModExp>(*(other.leftHandExpression));
				if (other.rightHandExpression != nullptr)
					rightHandExpression =
							std::make_unique<ModExp>(*(other.rightHandExpression));
				if (other.condition != nullptr)
					condition = std::make_unique<ModExp>(*(other.condition));
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
							std::make_unique<ModExp>(*(other.leftHandExpression));
				else
					leftHandExpression = nullptr;

				if (other.rightHandExpression != nullptr)
					rightHandExpression =
							std::make_unique<ModExp>(*(other.rightHandExpression));
				else
					rightHandExpression = nullptr;

				if (other.condition != nullptr)
					condition = std::make_unique<ModExp>(*(other.condition));
				else
					condition = nullptr;
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
			static bool deepEqual(ModExp* first, ModExp* second)
			{
				if (first == nullptr && second != nullptr)
					return false;
				if (first != nullptr && second == nullptr)
					return false;
				if (first == nullptr && second == nullptr)
					return true;
				return *first == *second;
			}
			ModExpKind kind{ ModExpKind::zero };
			std::unique_ptr<ModExp> leftHandExpression;
			std::unique_ptr<ModExp> rightHandExpression;
			std::unique_ptr<ModExp> condition;
		};

		/**
		 * Builds a reference expression based on the provided name
		 * of a var, and the type in which the values of that var will be casted.
		 */
		ModExp(
				std::string ref,
				BultinModTypes type,
				std::initializer_list<size_t> dimensions)
				: content(std::move(ref)), returnModType(type, std::move(dimensions))
		{
		}

		ModExp(std::string ref, ModType type)
				: content(std::move(ref)), returnModType(std::move(type))
		{
		}

		ModExp(std::string ref, BultinModTypes type)
				: content(std::move(ref)), returnModType(ModType(type))
		{
		}

		/**
		 * Builds a constant expression with the provided constant and the
		 * specified type in which the constant will be casted into. By default
		 * the type is deduced from the constant
		 */
		ModExp(ModConst constant, ModType returnModType)
				: content(std::move(constant)), returnModType(std::move(returnModType))
		{
		}
		ModExp(ModConst constant)
				: content(std::move(constant)),
					returnModType(getConstant().getModTypeOfLiteral())
		{
		}

		template<typename T>
		static ModExp constExp(T constant, ModType returnModType)
		{
			auto c = ModConst(constant);
			return ModExp(std::move(c), std::move(returnModType));
		}

		template<typename T>
		static ModExp constExp(T constant)
		{
			auto c = ModConst(constant);
			auto type = c.getModTypeOfLiteral();
			return ModExp(std::move(c), std::move(type));
		}

		/**
		 * Creates a call expression and it's result will be casted into the
		 * provided type
		 *
		 */
		ModExp(ModCall call, ModType returnType)
				: content(std::move(call)), returnModType(std::move(returnType))
		{
		}

		/**
		 * Creates a call expression
		 */
		ModExp(ModCall c): content(std::move(c)), returnModType(getCall().getType())
		{
			assert(getCall().getType().canBeCastedInto(returnModType));	 // NOLINT
		}

		ModExp& operator=(const ModExp& other) = default;
		/**
		 *We cannot default the move assigment operator because
		 * we might want to assign a node to the moved value of one of its own
		 *subexpressions. like exp = move(exp.getLeftHand()). If we default the
		 *operator exp would be deleated before the content of the children will be
		 *moved, therefore we must ensure we move out the content of the child
		 *before destorying the old value.
		 */
		ModExp& operator=(ModExp&& other)
		{
			if (this == &other)
				return *this;

			auto tempContent = std::move(other.content);
			auto tempModType = std::move(other.getModType());

			content = std::move(tempContent);
			returnModType = std::move(tempModType);
			return *this;
		}

		ModExp(const ModExp& other) = default;
		ModExp(ModExp&& other) = default;

		using iterator = ModChildrenIterator<ModExp>;
		using const_iterator = ModChildrenIterator<const ModExp>;
		[[nodiscard]] const_iterator begin() const { return const_iterator(*this); }
		[[nodiscard]] const_iterator end() const
		{
			return const_iterator(*this, childCount());
		}
		[[nodiscard]] iterator begin() { return iterator(*this); }
		[[nodiscard]] iterator end() { return iterator(*this, childCount()); }
		[[nodiscard]] llvm::iterator_range<iterator> range()
		{
			return llvm::make_range(begin(), end());
		}
		[[nodiscard]] llvm::iterator_range<const_iterator> range() const
		{
			return llvm::make_range(begin(), end());
		}

		/**
		 * \return true if it's not an operation or if every subexpression is
		 * compatible
		 */
		[[nodiscard]] bool areSubExpressionCompatibles() const
		{
			if (!isOperation())
				return true;

			if (isOperation<ModExpKind::at>())
				return true;

			if (isOperation<ModExpKind::module>() &&
					(getLeftHand().getModType().getBuiltin() != BultinModTypes::INT ||
					 getRightHand().getModType().getBuiltin() != BultinModTypes::INT))
				return false;

			auto compatible =
					getLeftHand().getModType().canBeCastedInto(getModType());

			if (!isBinary())
				return compatible;
			return compatible &&
						 getRightHand().getModType().canBeCastedInto(getModType());
		}

		[[nodiscard]] bool isReferenceAccess() const;
		[[nodiscard]] const std::string& getReferredVectorAccesss() const;
		[[nodiscard]] ModExp& getReferredVectorAccessExp();
		[[nodiscard]] const ModExp& getReferredVectorAccessExp() const;

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
		[[nodiscard]] static ModExp negate(ModExp exp)
		{
			return ModExp(
					ModExpKind::negate, std::make_unique<ModExp>(std::move(exp)));
		}

		[[nodiscard]] static ModExp index(ModExp exp)
		{
			assert(exp.getModType() == ModType(BultinModTypes::INT));	 // NOLINT
			return ModExp(
					ModExpKind::induction, std::make_unique<ModExp>(std::move(exp)));
		}

		/**
		 * \brief Creates a add expression based on two values.
		 * \warning Remember to move the two expression instead of copying them
		 * each time.
		 */
		[[nodiscard]] static ModExp add(ModExp lhs, ModExp rhs)
		{
			return ModExp(
					ModExpKind::add,
					std::make_unique<ModExp>(std::move(lhs)),
					std::make_unique<ModExp>(std::move(rhs)));
		}

		[[nodiscard]] static ModExp at(ModExp lhs, ModExp rhs)
		{
			assert(!lhs.getModType().isScalar());											 // NOLINT
			assert(rhs.getModType() == ModType(BultinModTypes::INT));	 // NOLINT
			auto type = lhs.getModType().sclidedType();
			return ModExp(
					ModExpKind::at,
					std::move(type),
					std::make_unique<ModExp>(std::move(lhs)),
					std::make_unique<ModExp>(std::move(rhs)));
		}

		/**
		 * \brief Creates a minus expression based on two values.
		 * \warning Remember to move the two expression instead of copying them
		 * each time.
		 */
		[[nodiscard]] static ModExp subtract(ModExp lhs, ModExp rhs)
		{
			return ModExp(
					ModExpKind::sub,
					std::make_unique<ModExp>(std::move(lhs)),
					std::make_unique<ModExp>(std::move(rhs)));
		}

		/**
		 * \brief Creates a multiply expression based on two values.
		 * \warning Remember to move the two expression instead of copying them
		 * each time.
		 */
		[[nodiscard]] static ModExp multiply(ModExp lhs, ModExp rhs)
		{
			return ModExp(
					ModExpKind::mult,
					std::make_unique<ModExp>(std::move(lhs)),
					std::make_unique<ModExp>(std::move(rhs)));
		}

		[[nodiscard]] static ModExp induction(ModExp lhs)
		{
			return ModExp(
					ModExpKind::induction, std::make_unique<ModExp>(std::move(lhs)));
		}

		/**
		 * \brief Creates a divisions expression based on two values.
		 * \warning Remember to move the two expression instead of copying them
		 * each time.
		 */
		[[nodiscard]] static ModExp divide(ModExp lhs, ModExp rhs)
		{
			return ModExp(
					ModExpKind::divide,
					std::make_unique<ModExp>(std::move(lhs)),
					std::make_unique<ModExp>(std::move(rhs)));
		}

		/**
		 * \brief Creates a powerOf expression based on two values.
		 * \warning Remember to move the two expression instead of copying them
		 * each time.
		 */
		[[nodiscard]] static ModExp elevate(ModExp lhs, ModExp rhs)
		{
			return ModExp(
					ModExpKind::elevation,
					std::make_unique<ModExp>(std::move(lhs)),
					std::make_unique<ModExp>(std::move(rhs)));
		}

		/**
		 * \brief Creates a modulo expression based on two values.
		 * \warning Remember to move the two expression instead of copying them
		 * each time.
		 */
		[[nodiscard]] static ModExp modulo(ModExp lhs, ModExp rhs)
		{
			return ModExp(
					ModExpKind::module,
					std::make_unique<ModExp>(std::move(lhs)),
					std::make_unique<ModExp>(std::move(rhs)));
		}

		/**
		 * \brief Creates a modulo expression based on two values.
		 * \warning Remember to move the two expression instead of copying them
		 * each time.
		 */
		[[nodiscard]] static ModExp cond(ModExp cond, ModExp lhs, ModExp rhs)
		{
			return ModExp(
					ModExpKind::conditional,
					std::make_unique<ModExp>(std::move(lhs)),
					std::make_unique<ModExp>(std::move(rhs)),
					std::make_unique<ModExp>(std::move(cond)));
		}

		/**
		 * Creates a greather than expression, the return type has the same
		 * dimensions as left hand value but of bools
		 */
		[[nodiscard]] static ModExp greaterThan(ModExp lhs, ModExp rhs)
		{
			auto type = lhs.getModType().as(BultinModTypes::BOOL);
			return ModExp(
					ModExpKind::greaterThan,
					std::move(type),
					std::make_unique<ModExp>(std::move(lhs)),
					std::make_unique<ModExp>(std::move(rhs)));
		}

		/**
		 * Creates a greather equal expression, the return type has the same
		 * dimensions as left hand value but of bools
		 */
		[[nodiscard]] static ModExp greaterEqual(ModExp lhs, ModExp rhs)
		{
			auto type = lhs.getModType().as(BultinModTypes::BOOL);
			return ModExp(
					ModExpKind::greaterEqual,
					std::move(type),
					std::make_unique<ModExp>(std::move(lhs)),
					std::make_unique<ModExp>(std::move(rhs)));
		}

		/**
		 * Creates a less equal expression, the return type has the same
		 * dimensions as left hand value but of bools
		 */
		[[nodiscard]] static ModExp lessEqual(ModExp lhs, ModExp rhs)
		{
			auto type = lhs.getModType().as(BultinModTypes::BOOL);
			return ModExp(
					ModExpKind::lessEqual,
					std::move(type),
					std::make_unique<ModExp>(std::move(lhs)),
					std::make_unique<ModExp>(std::move(rhs)));
		}
		/**
		 * Creates a less expression, the return type has the same
		 * dimensions as left hand value but of bools
		 */
		[[nodiscard]] static ModExp lessThan(ModExp lhs, ModExp rhs)
		{
			auto type = lhs.getModType().as(BultinModTypes::BOOL);
			return ModExp(
					ModExpKind::less,
					std::move(type),
					std::make_unique<ModExp>(std::move(lhs)),
					std::make_unique<ModExp>(std::move(rhs)));
		}

		/**
		 * Creates a equal expression, the return type has the same
		 * dimensions as left hand value but of bools
		 */
		[[nodiscard]] static ModExp equal(ModExp lhs, ModExp rhs)
		{
			auto type = lhs.getModType().as(BultinModTypes::BOOL);
			return ModExp(
					ModExpKind::equal,
					std::move(type),
					std::make_unique<ModExp>(std::move(lhs)),
					std::make_unique<ModExp>(std::move(rhs)));
		}

		/**
		 * Creates a different expression, the return type has the same
		 * dimensions as left hand value but of bools
		 */
		[[nodiscard]] static ModExp different(ModExp lhs, ModExp rhs)
		{
			auto type = lhs.getModType().as(BultinModTypes::BOOL);
			return ModExp(
					ModExpKind::different,
					std::move(type),
					std::make_unique<ModExp>(std::move(lhs)),
					std::make_unique<ModExp>(std::move(rhs)));
		}

		/**
		 * \brief Short hand for ModExp::greaterEqual
		 *
		 * \warning Notice that using the short hand will move the content no
		 * matter what
		 */
		[[nodiscard]] ModExp operator>=(const ModExp& other)
		{
			return ModExp::greaterThan(std::move(*this), std::move(other));
		}

		/**
		 * \brief Short hand for ModExp::greaterThan
		 *
		 * \warning Notice that using the short hand will move the content no
		 * matter what
		 */
		[[nodiscard]] ModExp operator>(const ModExp& other)
		{
			return ModExp::greaterThan(std::move(*this), std::move(other));
		}

		/**
		 * \brief Short hand for ModExp::lessThan
		 *
		 * \warning Notice that using the short hand will move the content no
		 * matter what
		 */
		[[nodiscard]] ModExp operator<(const ModExp& other)
		{
			return ModExp::lessThan(std::move(*this), std::move(other));
		}

		/**
		 * \brief Short hand for ModExp::lessEqual
		 *
		 * \warning Notice that using the short hand will move the content no
		 * matter what
		 */
		[[nodiscard]] ModExp operator<=(const ModExp& other)
		{
			return ModExp::lessEqual(std::move(*this), std::move(other));
		}

		/**
		 * \brief Short hand for ModExp::negate
		 *
		 * \warning Notice that using the short hand will move the content no
		 * matter what
		 */
		[[nodiscard]] ModExp operator!()
		{
			return ModExp::negate(std::move(*this));
		}

		/**
		 * \brief Short hand for ModExp::add
		 *
		 * \warning Notice that using the short hand will move the content no
		 * matter what.
		 */
		[[nodiscard]] ModExp operator+(const ModExp& other)
		{
			return ModExp::add(std::move(*this), std::move(other));
		}

		/**
		 * \brief Short hand for ModExp::subtract
		 *
		 * \warning Notice that using the short hand will move the content no
		 * matter what.
		 */
		[[nodiscard]] ModExp operator-(const ModExp& other)
		{
			return ModExp::subtract(std::move(*this), std::move(other));
		}

		/**
		 * \brief Short hand for ModExp::divide
		 *
		 * \warning Notice that using the short hand will move the content no
		 * matter what.
		 */
		[[nodiscard]] ModExp operator/(const ModExp& other)
		{
			return ModExp::divide(std::move(*this), std::move(other));
		}

		/**
		 * \brief Short hand for ModExp::multiply
		 *
		 * \warning Notice that using the short hand will move the content no
		 * matter what.
		 */
		[[nodiscard]] ModExp operator*(const ModExp& other)
		{
			return ModExp::multiply(std::move(*this), std::move(other));
		}

		/**
		 * \return true if the expression is a holding any kind of constant
		 */
		[[nodiscard]] bool isConstant() const
		{
			return !isOperation() && !isReference() && !isCall();
		}

		/**
		 * \return true if the expression is holding a call
		 *
		 */
		[[nodiscard]] bool isCall() const
		{
			return std::holds_alternative<ModCall>(content);
		}

		/**
		 * \return true if the expression is holding the expression of type C
		 */
		template<typename C>
		[[nodiscard]] bool isConstant() const
		{
			return std::holds_alternative<ModConst>(content) &&
						 getConstant().isA<C>();
		}

		/**
		 * \return true if the expression is holding an operation
		 */
		[[nodiscard]] bool isOperation() const
		{
			return std::holds_alternative<Operation>(content);
		}

		/**
		 *\return true if the expression is a operation of a particular kind
		 */
		template<ModExpKind kind>
		[[nodiscard]] bool isOperation() const
		{
			return isOperation() && getKind() == kind;
		}

		/**
		 *\return true if the expression is a operation of a particular kind
		 */
		[[nodiscard]] bool isOperation(ModExpKind kind) const
		{
			return isOperation() && getKind() == kind;
		}

		/**
		 * \pre isConstant<C>()
		 * \return the constant holded by this expression.
		 */
		[[nodiscard]] const ModConst& getConstant() const
		{
			assert(isConstant());	 // NOLINT
			return std::get<ModConst>(content);
		}

		/**
		 * \pre isConstant<C>()
		 * \return the constant holded by this expression.
		 */
		[[nodiscard]] ModConst& getConstant()
		{
			assert(isConstant());	 // NOLINT
			return std::get<ModConst>(content);
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

		void setReference(std::string name)
		{
			assert(isReference());
			content = std::move(name);
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

		[[nodiscard]] size_t childCount() const
		{
			if (isOperation())
				return getArity();
			if (isCall())
				return getCall().argsSize();
			return 0;
		}

		[[nodiscard]] ModExp& getChild(size_t index)
		{
			assert(index <= childCount());
			if (isCall())
				return getCall().at(index);
			if (isOperation())
			{
				switch (index)
				{
					case 0:
						return getLeftHand();
					case 1:
						return getRightHand();
					case 2:
						return getCondition();
				}
			}

			assert(false && "unreachable");
			return *this;
		}

		void setType(ModType tp)
		{
			if (isConstant())
				content = getConstant().as(tp.getBuiltin());
			returnModType = tp;
		}

		[[nodiscard]] const ModExp& getChild(size_t index) const
		{
			assert(index <= childCount());
			if (isCall())
				return getCall().at(index);
			if (isOperation())
			{
				switch (index)
				{
					case 0:
						return getLeftHand();
					case 1:
						return getRightHand();
					case 2:
						return getCondition();
				}
			}

			assert(false && "unreachable");
			return *this;
		}

		/**
		 * \pre isOperation()
		 *
		 * \return the type of the expression before being casted
		 * into the return type
		 */
		[[nodiscard]] ModType getOperationReturnType() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getOperationReturnType();
		}

		/**
		 * \pre isOperation()
		 * \return the left hand expression, or the only subexpression
		 * if it's a unary expression
		 */
		[[nodiscard]] const ModExp& getLeftHand() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getLeftHand();
		}

		/**
		 * \pre isOperation()
		 * \return the left hand expression, or the only subexpression
		 * if it's a unary expression
		 */
		[[nodiscard]] ModExp& getLeftHand()
		{
			assert(isOperation());	// NOLINT
			return getOperation().getLeftHand();
		}

		/**
		 * \pre isOperation()
		 * \return the right hand expression
		 */
		[[nodiscard]] const ModExp& getRightHand() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getRightHand();
		}

		/**
		 * \pre isOperation()
		 * \return the right hand expression
		 */
		[[nodiscard]] ModExp& getRightHand()
		{
			assert(isOperation());	// NOLINT
			return getOperation().getRightHand();
		}

		/**
		 * \pre isOperation()
		 * \return the condition of a conditional expression
		 */
		[[nodiscard]] ModExp& getCondition()
		{
			assert(isOperation());	// NOLINT
			return getOperation().getCondition();
		}

		/**
		 * \pre isOperation()
		 * \return the condition of a conditional expression
		 */
		[[nodiscard]] const ModExp& getCondition() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getCondition();
		}

		/**
		 * \pre isOperation()
		 * \return The kind of the expression
		 */
		[[nodiscard]] ModExpKind getKind() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getKind();
		}

		/**
		 * \return True iff the two expression are deeply equal, that is if every
		 * sub expression is equal to the other subexpressions.
		 */
		bool operator==(const ModExp& other) const
		{
			if (content != other.content)
				return false;
			return returnModType == other.returnModType;
		}

		/**
		 * \return The inverse of operator ==
		 */
		bool operator!=(const ModExp& other) const { return !(*this == other); }

		/**
		 * \return The type in which this expression will be casted into.
		 */
		[[nodiscard]] const ModType& getModType() const { return returnModType; }

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

		/**
		 * \req isCall()
		 * \return the call
		 */
		[[nodiscard]] const ModCall& getCall() const
		{
			assert(isCall());	 // NOLINT

			return std::get<ModCall>(content);
		}

		[[nodiscard]] ModCall& getCall()
		{
			assert(isCall());	 // NOLINT

			return std::get<ModCall>(content);
		}

		bool tryFoldConstant();

		ModExp(ModExpKind kind, ModType returnType, ModExp l, ModExp r)
				: content(Operation(
							kind,
							std::make_unique<ModExp>(std::move(l)),
							std::make_unique<ModExp>(std::move(r)),
							nullptr)),
					returnModType(std::move(returnType))
		{
			assert(!isOperation() || areSubExpressionCompatibles());
		}

		private:
		ModExp(
				ModExpKind kind,
				ModType retModType,
				std::unique_ptr<ModExp> lhs,
				std::unique_ptr<ModExp> rhs = nullptr,
				std::unique_ptr<ModExp> cond = nullptr)
				: content(
							Operation(kind, std::move(lhs), std::move(rhs), std::move(cond))),
					returnModType(std::move(retModType))
		{
			assert(!isOperation() || areSubExpressionCompatibles());
		}
		ModExp(
				ModExpKind kind,
				std::unique_ptr<ModExp> lhs,
				std::unique_ptr<ModExp> rhs = nullptr,
				std::unique_ptr<ModExp> cond = nullptr)
				: content(
							Operation(kind, std::move(lhs), std::move(rhs), std::move(cond))),
					returnModType(getLeftHand().getModType())
		{
			assert(!isOperation() || areSubExpressionCompatibles());	// NOLINT
		}

		ModExp(
				ModExpKind kind,
				llvm::SmallVector<std::unique_ptr<ModExp>, 3> subExp,
				ModType type)
				: content(Operation(kind, std::move(subExp))),
					returnModType(std::move(type))
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

		std::variant<Operation, ModConst, std::string, ModCall> content;
		ModType returnModType;
	};	// namespace modelica

	/**
	 * Visitor is class that must implement void visit(ModExp&) and
	 * void afterVisit(ModExp&). visit will be called for every sub expression
	 * of exp (exp included) if top down order. AfterVisit will be invoked after
	 * each children has been visited.
	 *
	 * You can implement a empty afterVisit to obtain a topDown visitor and not
	 * implement visit to get a bottomUpVisitor.
	 */
	template<typename ModExp, typename Visitor>
	void visit(ModExp& exp, Visitor& visitor)
	{
		visitor.visit(exp);
		for (ModExp& child : exp)
			visit(child, visitor);
		visitor.afterVisit(exp);
	}

}	 // namespace modelica
