#pragma once

#include <vector>

#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"
#include "modelica/AST/Expr.hpp"
#include "modelica/utils/SourceRange.hpp"

namespace modelica
{
	class Statement
	{
		public:
		using ExprIterator = vectorUnique<Expr>::iterator;
		using ConstExprIterator = vectorUnique<Expr>::const_iterator;

		enum StatemenKind
		{
			CallStatement,
			AssignStatement,
			BreakStatement,
			ReturnStatement,
			CompositeStatement,
			IfStatement,
			WhenStatement,
			WhileStatement,
			ForStatement,
			LastCompositeStatement,
			LastStatement
		};

		Statement(
				SourceRange location, StatemenKind kind, vectorUnique<Expr> exprs = {})
				: kind(kind), loc(location), expressions(std::move(exprs))
		{
		}
		virtual ~Statement() = default;

		[[nodiscard]] StatemenKind getKind() const { return kind; }
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

		protected:
		[[nodiscard]] const vectorUnique<Expr>& getExpressions() const
		{
			return expressions;
		}
		[[nodiscard]] vectorUnique<Expr>& getExpressions() { return expressions; }

		private:
		StatemenKind kind;
		SourceRange loc;
		vectorUnique<Expr> expressions;
	};

	/**
	 * This is the template used by every ast leaf member as an alias
	 * to implement classof, which is used by llvm::cast
	 */
	template<Statement::StatemenKind kind>
	constexpr bool leafClassOf(const Statement* e)
	{
		return e->getKind() == kind;
	}

	/**
	 * This is the template used by every ast non leaf member as an alias
	 * to implement classof, which is used by llvm::cast
	 */
	template<Statement::StatemenKind kind, Statement::StatemenKind lastKind>
	constexpr bool nonLeafClassOf(const Statement* e)
	{
		return e->getKind() >= kind && e->getKind() < lastKind;
	}
	using UniqueStmt = std::unique_ptr<Statement>;

	/**
	 * Rappresents a statement that is composed by multiples sub equations, such
	 * as a when equation.
	 */
	class CompositeStatement: public Statement
	{
		using StmtIterator = vectorUnique<Statement>::iterator;
		using ConstStmtIterator = vectorUnique<Statement>::const_iterator;

		public:
		CompositeStatement(
				SourceRange loc,
				StatemenKind kind = StatemenKind::CompositeStatement,
				vectorUnique<Statement> children = {},
				vectorUnique<Expr> exprs = {})
				: Statement(loc, kind, std::move(exprs)), equations(std::move(children))
		{
		}
		CompositeStatement(SourceRange loc, vectorUnique<Statement> children)
				: Statement(loc, StatemenKind::CompositeStatement),
					equations(std::move(children))
		{
		}
		~CompositeStatement() override = default;
		[[nodiscard]] int eqSize() const { return equations.size(); }
		[[nodiscard]] StmtIterator stmtBegin() { return equations.begin(); }
		[[nodiscard]] StmtIterator stmtEnd() { return equations.end(); }
		[[nodiscard]] ConstStmtIterator stmtCbegin() const
		{
			return equations.cbegin();
		}
		[[nodiscard]] ConstStmtIterator stmtCend() const
		{
			return equations.cend();
		}

		static constexpr auto classof = nonLeafClassOf<
				StatemenKind::CompositeStatement,
				StatemenKind::LastCompositeStatement>;

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		void removeNullEq()
		{
			equations.erase(std::remove(stmtBegin(), stmtEnd(), nullptr), stmtEnd());
		}

		protected:
		[[nodiscard]] const vectorUnique<Statement>& getStatements() const
		{
			return equations;
		}
		[[nodiscard]] vectorUnique<Statement>& getStatements() { return equations; }

		private:
		vectorUnique<Statement> equations;
	};

	class IfStatement: public CompositeStatement
	{
		public:
		IfStatement(
				SourceRange loc,
				vectorUnique<Expr> exprs,
				vectorUnique<Statement> statments)
				: CompositeStatement(
							loc,
							StatemenKind::IfStatement,
							std::move(statments),
							std::move(exprs))
		{
		}

		[[nodiscard]] unsigned branchesSize() const
		{
			return getStatements().size();
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
		[[nodiscard]] const Statement* getStatement(unsigned index)
		{
			if (index >= getStatements().size())
				return nullptr;
			return getStatements()[index].get();
		}
		[[nodiscard]] const Expr* getElseBranch()
		{
			if (!hasFinalElse())
				return nullptr;
			return getExpressions().back().get();
		}

		static constexpr auto classof = leafClassOf<StatemenKind::IfStatement>;

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
	};

	class WhileStatement: public CompositeStatement
	{
		public:
		WhileStatement(
				SourceRange loc, UniqueExpr expr, vectorUnique<Statement> statement)
				: CompositeStatement(loc, StatemenKind::WhileStatement, move(statement))
		{
			getExpressions().push_back(std::move(expr));
		}

		~WhileStatement() override = default;

		static constexpr auto classof = leafClassOf<StatemenKind::WhileStatement>;

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		[[nodiscard]] const Expr* getCondition()
		{
			return getExpressions()[0].get();
		}
		[[nodiscard]] const Statement* getStatement()
		{
			return getStatements()[0].get();
		}
	};

	class WhenStatement: public CompositeStatement
	{
		public:
		WhenStatement(
				SourceRange loc,
				vectorUnique<Expr> exprs,
				vectorUnique<Statement> statements)
				: CompositeStatement(
							loc,
							StatemenKind::WhenStatement,
							std::move(statements),
							std::move(exprs))
		{
		}

		~WhenStatement() override = default;

		static constexpr auto classof = leafClassOf<StatemenKind::WhenStatement>;

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		[[nodiscard]] unsigned branchesSize() const
		{
			return getStatements().size();
		}
		[[nodiscard]] const Expr* getCondition(unsigned index)
		{
			if (index >= getExpressions().size())
				return nullptr;
			return getExpressions()[index].get();
		}
		[[nodiscard]] const Statement* getEquation(unsigned index)
		{
			if (index >= getStatements().size())
				return nullptr;
			return getStatements()[index].get();
		}
	};

	class ForStatement: public CompositeStatement
	{
		public:
		ForStatement(
				SourceRange loc,
				vectorUnique<Expr> forExpr,
				vectorUnique<Statement> statements,
				std::vector<std::string> names)
				: CompositeStatement(
							loc, StatemenKind::ForStatement, move(statements), move(forExpr)),
					names(std::move(names))
		{
		}
		~ForStatement() override = default;
		static constexpr auto classof = leafClassOf<StatemenKind::ForStatement>;

		[[nodiscard]] unsigned statementCount() const
		{
			return getStatements().size();
		}
		[[nodiscard]] const Statement* getStatement(unsigned index)
		{
			if (index >= statementCount())
				return nullptr;
			return getStatements()[index].get();
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

	class AssignStatement: public Statement
	{
		public:
		AssignStatement(
				SourceRange loc,
				std::unique_ptr<Expr> leftHand,
				std::unique_ptr<Expr> rightHand)
				: Statement(loc, StatemenKind::AssignStatement)
		{
			getExpressions().emplace_back(std::move(leftHand));
			getExpressions().emplace_back(std::move(rightHand));
		}
		~AssignStatement() override = default;
		static constexpr auto classof = leafClassOf<StatemenKind::AssignStatement>;

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

	class CallStatement: public Statement
	{
		public:
		CallStatement(SourceRange loc, std::unique_ptr<Expr> callExpression)
				: Statement(loc, StatemenKind::CallStatement)
		{
			getExpressions().emplace_back(std::move(callExpression));
		}
		~CallStatement() override = default;
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}

		static constexpr auto classof = leafClassOf<StatemenKind::CallStatement>;

		[[nodiscard]] const Expr* getCallExpr() const
		{
			return getExpressions()[0].get();
		}
	};

	class BreakStatement: public Statement
	{
		public:
		BreakStatement(SourceRange loc)
				: Statement(loc, StatemenKind::BreakStatement)
		{
		}
		~BreakStatement() override = default;
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}

		static constexpr auto classof = leafClassOf<StatemenKind::BreakStatement>;
	};

	class ReturnStatement: public Statement
	{
		public:
		ReturnStatement(SourceRange loc)
				: Statement(loc, StatemenKind::ReturnStatement)
		{
		}
		~ReturnStatement() override = default;
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}

		static constexpr auto classof = leafClassOf<StatemenKind::ReturnStatement>;
	};
}	// namespace modelica
