#pragma once

#include <vector>

#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"
#include "modelica/AST/Expr.hpp"
#include "modelica/utils/SourceRange.hpp"

namespace modelica
{
	class Equation
	{
		public:
		using ExprIterator = vectorUnique<Expr>::iterator;
		using ConstExprIterator = vectorUnique<Expr>::const_iterator;

		enum EquationKind
		{
			SimpleEquation,
			TerminateEquation,
			AssertEquation,
			CompositeEquation,
			IfEquation,
			ForEquation,
			ConnectClause,
			WhenEquation,
			LastCompositeEquation,
			LastEquation
		};

		Equation(
				SourceRange location, EquationKind kind, vectorUnique<Expr> exprs = {})
				: kind(kind), loc(location), expressions(std::move(exprs))
		{
		}
		virtual ~Equation() = default;

		[[nodiscard]] EquationKind getKind() const { return kind; }
		[[nodiscard]] const SourceRange& getRange() const { return loc; }

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
	};

	using UniqueEq = std::unique_ptr<Equation>;

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

		static bool classof(const Equation* e)
		{
			return e->getKind() >= EquationKind::CompositeEquation &&
						 e->getKind() < EquationKind::LastCompositeEquation;
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
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
		static bool classof(const Equation* e)
		{
			return e->getKind() == EquationKind::IfEquation;
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
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
		static bool classof(const Equation* e)
		{
			return e->getKind() == EquationKind::ForEquation;
		}
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
		static bool classof(const Equation* e)
		{
			return e->getKind() == EquationKind::SimpleEquation;
		}

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

}	// namespace modelica
