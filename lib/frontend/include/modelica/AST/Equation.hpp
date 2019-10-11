#pragma once

#include <vector>

#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"
#include "modelica/AST/Expr.hpp"
#include "modelica/utils/SourceRange.hpp"

namespace modelica
{
	/**
	 * Read the llvm programmer manual in the section regarding how to build a
	 * hierarchy This is built exactly in that wasy, except we have virtual
	 * desctructors because we are not just allocating everything in a custom
	 * allocator and never destryoing them.
	 *
	 * Forgetting to add a virtual destructor will cause undefined behaviour and
	 * probaly memory leaks.
	 *
	 * Remember to add a custo isChoerent to every class so that we can
	 * perform integirty checks after every visit.
	 */
	class Equation
	{
		public:
		using ExprIterator = vectorUnique<Expr>::iterator;
		using ConstExprIterator = vectorUnique<Expr>::const_iterator;

		enum EquationKind
		{
			SimpleEquation,
			ConnectClause,
			CallEquation,
			CompositeEquation,
			IfEquation,
			ForEquation,
			WhenEquation,
			LastCompositeEquation,
			LastEquation
		};

		Equation(
				SourceRange location,
				EquationKind kind,
				vectorUnique<Expr> exprs = {},
				std::string cmnt = "")
				: kind(kind),
					loc(location),
					expressions(std::move(exprs)),
					comment(std::move(cmnt))
		{
		}
		virtual ~Equation() = default;

		[[nodiscard]] EquationKind getKind() const { return kind; }
		[[nodiscard]] const SourceRange& getRange() const { return loc; }

		/**
		 * Return the number of subexpressions directly child of the
		 * current equation.
		 */
		[[nodiscard]] int exprSize() const { return expressions.size(); }
		[[nodiscard]] ExprIterator exprBegin() { return expressions.begin(); }
		[[nodiscard]] ExprIterator exprEnd() { return expressions.end(); }
		[[nodiscard]] ConstExprIterator exprCbegin() const
		{
			return expressions.cbegin();
		}
		[[nodiscard]] ConstExprIterator exprCend() const
		{
			return expressions.cend();
		}

		/**
		 * just a wrapper around the erase remove idiom
		 */
		void removeNullExpr()
		{
			expressions.erase(
					std::remove(exprBegin(), exprEnd(), nullptr), exprEnd());
		}

		[[nodiscard]] const std::string& getComment() const { return comment; }
		void setComment(std::string newComment) { comment = std::move(newComment); }

		protected:
		[[nodiscard]] const vectorUnique<Expr>& getExpressions() const
		{
			return expressions;
		}
		[[nodiscard]] vectorUnique<Expr>& getExpressions() { return expressions; }

		private:
		EquationKind kind;
		SourceRange loc;
		vectorUnique<Expr> expressions;
		std::string comment;
	};

	/**
	 * This is the template used by every ast leaf member as an alias
	 * to implement classof, which is used by llvm::cast
	 */
	template<Equation::EquationKind kind>
	constexpr bool leafClassOf(const Equation* e)
	{
		return e->getKind() == kind;
	}

	/**
	 * This is the template used by every ast non leaf member as an alias
	 * to implement classof, which is used by llvm::cast
	 */
	template<Equation::EquationKind kind, Equation::EquationKind lastKind>
	constexpr bool nonLeafClassOf(const Equation* e)
	{
		return e->getKind() >= kind && e->getKind() < lastKind;
	}
	using UniqueEq = std::unique_ptr<Equation>;

	/**
	 * Rappresents a euqation that is composed by multiples sub equations, such as
	 * a when equation.
	 */
	class CompositeEquation: public Equation
	{
		using EqIterator = vectorUnique<Equation>::iterator;
		using ConstEqIterator = vectorUnique<Equation>::const_iterator;

		public:
		CompositeEquation(
				SourceRange loc,
				EquationKind kind = EquationKind::CompositeEquation,
				vectorUnique<Equation> children = {},
				vectorUnique<Expr> exprs = {})
				: Equation(loc, kind, std::move(exprs)), equations(std::move(children))
		{
		}
		CompositeEquation(SourceRange loc, vectorUnique<Equation> children)
				: Equation(loc, EquationKind::CompositeEquation),
					equations(std::move(children))
		{
		}
		~CompositeEquation() override = default;
		[[nodiscard]] int eqSize() const { return equations.size(); }
		[[nodiscard]] EqIterator eqBegin() { return equations.begin(); }
		[[nodiscard]] EqIterator eqEnd() { return equations.end(); }
		[[nodiscard]] ConstEqIterator eqCbegin() const
		{
			return equations.cbegin();
		}
		[[nodiscard]] ConstEqIterator eqCend() const { return equations.cend(); }

		static constexpr auto classof = nonLeafClassOf<
				EquationKind::CompositeEquation,
				EquationKind::LastCompositeEquation>;

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		void removeNullEq()
		{
			equations.erase(std::remove(eqBegin(), eqEnd(), nullptr), eqEnd());
		}

		protected:
		[[nodiscard]] const vectorUnique<Equation>& getEquations() const
		{
			return equations;
		}
		[[nodiscard]] vectorUnique<Equation>& getEquations() { return equations; }

		private:
		vectorUnique<Equation> equations;
	};

	class IfEquation: public CompositeEquation
	{
		public:
		IfEquation(
				SourceRange loc, vectorUnique<Expr> exprs, vectorUnique<Equation> equs)
				: CompositeEquation(
							loc, EquationKind::IfEquation, std::move(equs), std::move(exprs))
		{
		}

		[[nodiscard]] unsigned branchesSize() const
		{
			return getEquations().size();
		}
		[[nodiscard]] bool hasFinalElse() const
		{
			return getExpressions().size() < branchesSize();
		}
		[[nodiscard]] const Expr* getCondition(unsigned index)
		{
			if (index >= getExpressions().size())
				return nullptr;
			return getExpressions()[index].get();
		}
		[[nodiscard]] const Equation* getEquation(unsigned index)
		{
			if (index >= getEquations().size())
				return nullptr;
			return getEquations()[index].get();
		}
		[[nodiscard]] const Expr* getElseBranch()
		{
			if (!hasFinalElse())
				return nullptr;
			return getExpressions().back().get();
		}

		static constexpr auto classof = leafClassOf<EquationKind::IfEquation>;

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
	};

	class WhenEquation: public CompositeEquation
	{
		public:
		WhenEquation(
				SourceRange loc, vectorUnique<Expr> exprs, vectorUnique<Equation> equs)
				: CompositeEquation(
							loc,
							EquationKind::WhenEquation,
							std::move(equs),
							std::move(exprs))
		{
		}

		~WhenEquation() override = default;

		static constexpr auto classof = leafClassOf<EquationKind::WhenEquation>;

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		[[nodiscard]] unsigned branchesSize() const
		{
			return getEquations().size();
		}
		[[nodiscard]] const Expr* getCondition(unsigned index)
		{
			if (index >= getExpressions().size())
				return nullptr;
			return getExpressions()[index].get();
		}
		[[nodiscard]] const Equation* getEquation(unsigned index)
		{
			if (index >= getEquations().size())
				return nullptr;
			return getEquations()[index].get();
		}
	};

	class ForEquation: public CompositeEquation
	{
		public:
		ForEquation(
				SourceRange loc,
				vectorUnique<Expr> forExpr,
				vectorUnique<Equation> equations,
				std::vector<std::string> names)
				: CompositeEquation(
							loc, EquationKind::ForEquation, move(equations), move(forExpr)),
					names(std::move(names))
		{
		}
		~ForEquation() override = default;
		static constexpr auto classof = leafClassOf<EquationKind::ForEquation>;

		[[nodiscard]] unsigned equationsCount() const
		{
			return getEquations().size();
		}
		[[nodiscard]] const Equation* getEquation(unsigned index)
		{
			if (index >= equationsCount())
				return nullptr;
			return getEquations()[index].get();
		}

		[[nodiscard]] unsigned forExpressionsCount() const
		{
			return getExpressions().size();
		}
		[[nodiscard]] const Expr* getForExpression(unsigned index)
		{
			if (index >= forExpressionsCount())
				return nullptr;
			return getExpressions()[index].get();
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}

		private:
		std::vector<std::string> names;
	};

	class SimpleEquation: public Equation
	{
		public:
		SimpleEquation(
				SourceRange loc,
				std::unique_ptr<Expr> leftHand,
				std::unique_ptr<Expr> rightHand)
				: Equation(loc, EquationKind::SimpleEquation)
		{
			getExpressions().emplace_back(std::move(leftHand));
			getExpressions().emplace_back(std::move(rightHand));
		}
		~SimpleEquation() override = default;
		static constexpr auto classof = leafClassOf<EquationKind::SimpleEquation>;

		[[nodiscard]] const Expr* getLeftHand() const
		{
			return getExpressions()[0].get();
		}
		[[nodiscard]] const Expr* getRightHand() const
		{
			return getExpressions()[1].get();
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
	};

	class CallEquation: public Equation
	{
		public:
		CallEquation(SourceRange loc, std::unique_ptr<Expr> callExpression)
				: Equation(loc, EquationKind::CallEquation)
		{
			getExpressions().emplace_back(std::move(callExpression));
		}
		~CallEquation() override = default;
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}

		static constexpr auto classof = leafClassOf<EquationKind::CallEquation>;

		[[nodiscard]] const Expr* getCallExpr() const
		{
			return getExpressions()[0].get();
		}
	};

	class ConnectClause: public Equation
	{
		public:
		ConnectClause(
				SourceRange loc,
				std::unique_ptr<Expr> firstParam,
				std::unique_ptr<Expr> secondParam)
				: Equation(loc, EquationKind::ConnectClause)
		{
			getExpressions().emplace_back(std::move(firstParam));
			getExpressions().emplace_back(std::move(secondParam));
		}
		~ConnectClause() override = default;
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}

		static constexpr auto classof = leafClassOf<EquationKind::ConnectClause>;

		[[nodiscard]] const Expr* getFirstParam() const
		{
			return getExpressions()[0].get();
		}
		[[nodiscard]] const Expr* getSecondParam() const
		{
			return getExpressions()[1].get();
		}
	};

}	// namespace modelica
