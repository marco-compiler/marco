#pragma once

#include <cassert>
#include <memory>
#include <variant>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/lowerer/SimConst.hpp"
#include "modelica/lowerer/SimType.hpp"

namespace modelica
{
	enum class SimExpKind
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

			[[nodiscard]] bool isUnary() const
			{
				return kind >= SimExpKind::negate && kind <= SimExpKind::negate;
			}

			[[nodiscard]] bool isBinary() const
			{
				return kind >= SimExpKind::add && kind <= SimExpKind::module;
			}

			[[nodiscard]] bool isTernary() const
			{
				return kind >= SimExpKind::conditional &&
							 kind <= SimExpKind::conditional;
			}

			[[nodiscard]] const SimExp& getLeftHand() const
			{
				assert(isUnary() || isBinary() || isTernary());	// NOLINT
				return *leftHandExpression;
			}

			[[nodiscard]] SimExp& getLeftHand()
			{
				assert(isUnary() || isBinary() || isTernary());	// NOLINT
				return *leftHandExpression;
			}

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
							std::make_unique<SimExp>(*(other.leftHandExpression));
				if (other.rightHandExpression != nullptr)
					rightHandExpression =
							std::make_unique<SimExp>(*(other.rightHandExpression));

				if (other.condition != nullptr)
					condition = std::make_unique<SimExp>(*(other.condition));
			}

			Operation& operator=(const Operation& other)
			{
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
			bool deepEqual(SimExp* first, SimExp* second) const
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

		template<typename... T>
		SimExp(std::string ref, BultinSimTypes type, T... dimensions)
				: content(std::move(ref)), returnSimType(type, dimensions...)
		{
		}

		template<typename C>
		SimExp(
				SimConst<C> constant,
				SimType returnSimType = SimType(typeToBuiltin<C>()))
				: content(std::move(constant)), returnSimType(std::move(returnSimType))
		{
		}
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		[[nodiscard]] static SimExp negate(SimExp exp)
		{
			return SimExp(
					SimExpKind::negate, std::make_unique<SimExp>(std::move(exp)));
		}

		[[nodiscard]] static SimExp add(SimExp lhs, SimExp rhs)
		{
			return SimExp(
					SimExpKind::add,
					std::make_unique<SimExp>(std::move(lhs)),
					std::make_unique<SimExp>(std::move(rhs)));
		}

		[[nodiscard]] static SimExp subtract(SimExp lhs, SimExp rhs)
		{
			return SimExp(
					SimExpKind::sub,
					std::make_unique<SimExp>(std::move(lhs)),
					std::make_unique<SimExp>(std::move(rhs)));
		}

		[[nodiscard]] static SimExp multiply(SimExp lhs, SimExp rhs)
		{
			return SimExp(
					SimExpKind::mult,
					std::make_unique<SimExp>(std::move(lhs)),
					std::make_unique<SimExp>(std::move(rhs)));
		}

		[[nodiscard]] static SimExp divide(SimExp lhs, SimExp rhs)
		{
			return SimExp(
					SimExpKind::divide,
					std::make_unique<SimExp>(std::move(lhs)),
					std::make_unique<SimExp>(std::move(rhs)));
		}

		[[nodiscard]] SimExp operator!()
		{
			return SimExp::negate(std::move(*this));
		}

		[[nodiscard]] SimExp operator+(const SimExp& other)
		{
			return SimExp::add(std::move(*this), std::move(other));
		}

		[[nodiscard]] SimExp operator-(const SimExp& other)
		{
			return SimExp::subtract(std::move(*this), std::move(other));
		}

		[[nodiscard]] SimExp operator/(const SimExp& other)
		{
			return SimExp::divide(std::move(*this), std::move(other));
		}

		[[nodiscard]] bool isConstant() const
		{
			return !isOperation() && !isReference();
		}

		template<typename C>
		[[nodiscard]] bool isConstant() const
		{
			return std::holds_alternative<SimConst<C>>(content);
		}

		[[nodiscard]] bool isOperation() const
		{
			return std::holds_alternative<Operation>(content);
		}

		template<typename C>
		[[nodiscard]] const SimConst<C>& getConstant() const
		{
			assert(isConstant<C>());	// NOLINT
			return std::get<SimConst<C>>(content);
		}

		template<typename C>
		[[nodiscard]] SimConst<C>& getConstant()
		{
			assert(isConstant<C>());	// NOLINT
			return std::get<SimConst<C>>(content);
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

		[[nodiscard]] const SimExp& getLeftHand() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getLeftHand();
		}

		[[nodiscard]] SimExp& getLeftHand()
		{
			assert(isOperation());	// NOLINT
			return getOperation().getLeftHand();
		}

		[[nodiscard]] const SimExp& getRightHand() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getRightHand();
		}

		[[nodiscard]] SimExp& getRightHand()
		{
			assert(isOperation());	// NOLINT
			return getOperation().getRightHand();
		}

		[[nodiscard]] SimExp& getCondition()
		{
			assert(isOperation());	// NOLINT
			return getOperation().getCondition();
		}

		[[nodiscard]] const SimExp& getCondition() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getCondition();
		}

		[[nodiscard]] SimExpKind getKind() const
		{
			assert(isOperation());	// NOLINT
			return getOperation().getKind();
		}

		bool operator==(const SimExp& other) const
		{
			if (content != other.content)
				return false;
			return returnSimType == other.returnSimType;
		}
		bool operator!=(const SimExp& other) const { return !(*this == other); }

		[[nodiscard]] const SimType& getSimType() const { return returnSimType; }

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

}	// namespace modelica
