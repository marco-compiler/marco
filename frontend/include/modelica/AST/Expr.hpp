#pragma once

#include <cassert>
#include <string>
#include <vector>

#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "modelica/AST/Type.hpp"
#include "modelica/utils/SourceRange.hpp"

namespace modelica
{
	template<typename T>
	using vectorUnique = std::vector<std::unique_ptr<T>>;

	class Expr
	{
		public:
		enum ExprKind
		{
			BoolLiteralExpr,
			IntLiteralExpr,
			StringLiteralExpr,
			FloatLiteralExpr,
			ExpressionList,
			BinaryExpr,
			LastBinaryExpr,
			UnaryExpr,
			LastUnaryExpr,
			IfElseExpr,
			ArrayConcatExpression,
			RangeExpression,
			ArrayConstructorExpression,
			DirectArrayConstructorExpression,
			ForInArrayConstructorExpression,
			LastArrayConstructorExpression,
			LastExpressionList

		};

		Expr(SourceRange location, Type type, ExprKind kind)
				: location(std::move(location)), kind(kind), type(std::move(type))
		{
		}
		virtual ~Expr() = default;

		[[nodiscard]] const SourceRange& getRange() const { return location; }
		[[nodiscard]] ExprKind getKind() const { return kind; }
		[[nodiscard]] const Type& getType() const { return type; }

		private:
		const SourceRange location;
		const ExprKind kind;
		const Type type;
	};

	using UniqueExpr = std::unique_ptr<Expr>;
	using ExprIterator = vectorUnique<Expr>::iterator;
	using ConstExprIterator = vectorUnique<Expr>::const_iterator;

	enum class BinaryExprOp
	{
		Sum,
		Subtraction,
		Multiply,
		Division,
		PowerOf,
		LogicalOr,
		LogicalAnd,
		Less,
		LessEqual,
		Greater,
		GreatureEqual,
		Equal,
		Different
	};

	class ExprList: public Expr
	{
		public:
		ExprList(
				SourceRange location,
				Type type = Type(BuiltinType::None),
				ExprKind kind = ExprKind::ExpressionList,
				std::vector<UniqueExpr> exprs = {})
				: Expr(std::move(location), type, kind), expressions(std::move(exprs))
		{
		}
		ExprList(SourceRange location, std::vector<UniqueExpr> exprs)
				: Expr(
							std::move(location),
							Type(BuiltinType::None),
							ExprKind::ExpressionList),
					expressions(std::move(exprs))
		{
		}
		static bool classof(const Expr* e)
		{
			return e->getKind() >= ExprKind::ExpressionList &&
						 e->getKind() < ExprKind::LastExpressionList;
		}
		[[nodiscard]] llvm::Error isConsistent() const;
		[[nodiscard]] int size() const { return expressions.size(); }
		[[nodiscard]] ExprIterator begin() { return expressions.begin(); }
		[[nodiscard]] ExprIterator end() { return expressions.end(); }
		[[nodiscard]] ConstExprIterator cbegin() const
		{
			return expressions.cbegin();
		}
		[[nodiscard]] ConstExprIterator cend() const { return expressions.cend(); }
		[[nodiscard]] Expr* at(int index) { return expressions.at(index).get(); }
		[[nodiscard]] const Expr* at(int index) const
		{
			return expressions.at(index).get();
		}
		[[nodiscard]] vectorUnique<Expr> takeVector()
		{
			return std::move(expressions);
		}
		[[nodiscard]] llvm::iterator_range<ExprIterator> children()
		{
			return llvm::make_range(begin(), end());
		}
		[[nodiscard]] llvm::iterator_range<ConstExprIterator> children() const
		{
			return llvm::make_range(cbegin(), cend());
		}

		void emplace(UniqueExpr expr) { expressions.emplace_back(std::move(expr)); }
		~ExprList() override = default;

		protected:
		[[nodiscard]] vectorUnique<Expr>& getExpressions() { return expressions; }
		[[nodiscard]] const vectorUnique<Expr>& getExpressions() const
		{
			return expressions;
		}

		private:
		vectorUnique<Expr> expressions;
	};

	class BinaryExpr: public ExprList
	{
		public:
		BinaryExpr(
				SourceRange location,
				BinaryExprOp op,
				UniqueExpr lhs,
				UniqueExpr rhs,
				ExprKind kind = ExprKind::BinaryExpr)
				: ExprList(location, Type(BuiltinType::Unknown), kind), operation(op)
		{
			getExpressions().emplace_back(std::move(lhs));
			getExpressions().emplace_back(std::move(rhs));
		}

		[[nodiscard]] const Expr* getLeftHand() const { return at(0); }
		[[nodiscard]] const Expr* getRightHand() const { return at(1); }

		static bool classof(const Expr* e)
		{
			return e->getKind() >= ExprKind::BinaryExpr &&
						 e->getKind() < ExprKind::LastBinaryExpr;
		}
		[[nodiscard]] BinaryExprOp getOpCode() const { return operation; }
		void setOpCode(BinaryExprOp op) { operation = op; }

		[[nodiscard]] llvm::Error isConsistent() const;
		~BinaryExpr() final = default;

		private:
		BinaryExprOp operation;
	};

	enum class UnaryExprOp
	{
		LogicalNot,
		Plus,
		Minus
	};

	class UnaryExpr: public ExprList
	{
		public:
		UnaryExpr(
				SourceRange location,
				UnaryExprOp op,
				UniqueExpr oprnd,
				ExprKind kind = ExprKind::UnaryExpr)
				: ExprList(std::move(location), Type(BuiltinType::Unknown), kind),
					operation(op)
		{
			getExpressions().emplace_back(std::move(oprnd));
		}

		static bool classof(const Expr* e)
		{
			return e->getKind() >= ExprKind::UnaryExpr &&
						 e->getKind() < ExprKind::LastUnaryExpr;
		}
		[[nodiscard]] UnaryExprOp getOpCode() const { return operation; }
		void setOpCode(UnaryExprOp op) { operation = op; }
		[[nodiscard]] const Expr* getOperand() const { return at(0); }

		[[nodiscard]] llvm::Error isConsistent() const;
		~UnaryExpr() final = default;

		private:
		UnaryExprOp operation;
	};

	class RangeExpr: public ExprList
	{
		public:
		RangeExpr(
				SourceRange loc,
				UniqueExpr start,
				UniqueExpr stop,
				UniqueExpr step = nullptr)
				: ExprList(loc, Type(BuiltinType::Unknown), ExprKind::RangeExpression)
		{
			getExpressions().emplace_back(std::move(start));
			getExpressions().emplace_back(std::move(stop));
			if (step.get() != nullptr)
				getExpressions().emplace_back(move(step));
		}

		static bool classof(const Expr* e)
		{
			return e->getKind() == ExprKind::RangeExpression;
		}

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		[[nodiscard]] const Expr* getStart() const { return at(0); }
		[[nodiscard]] const Expr* getStop() const { return at(1); }
		[[nodiscard]] bool hasStep() const { return size() == 3; }
		[[nodiscard]] const Expr* getStep() const
		{
			if (!hasStep())
				return nullptr;
			return at(2);
		}
		~RangeExpr() final = default;
	};
	class ArrayConstructorExpr: public ExprList
	{
		public:
		ArrayConstructorExpr(
				SourceRange loc, vectorUnique<Expr> children, ExprKind kind)
				: ExprList(loc, Type(BuiltinType::Unknown), kind, std::move(children))
		{
		}
		static bool classof(const Expr* e)
		{
			return e->getKind() >= ExprKind::ArrayConstructorExpression &&
						 e->getKind() < ExprKind::LastArrayConstructorExpression;
		}
		~ArrayConstructorExpr() override = default;
	};

	class ForInArrayConstructorExpr: public ArrayConstructorExpr
	{
		public:
		ForInArrayConstructorExpr(
				SourceRange loc,
				UniqueExpr evalExp,
				vectorUnique<Expr> exprPairs,
				std::vector<std::string> names)
				: ArrayConstructorExpr(
							loc,
							std::move(exprPairs),
							ExprKind::ForInArrayConstructorExpression),
					names(std::move(names))
		{
			getExpressions().emplace_back(std::move(evalExp));
		}

		[[nodiscard]] const Expr* getEvaluationExpr() const
		{
			return getExpressions().back().get();
		}
		[[nodiscard]] const std::string& getDeclaredName(int index)
		{
			return names[index];
		}

		static bool classof(const Expr* e)
		{
			return e->getKind() == ExprKind::ForInArrayConstructorExpression;
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~ForInArrayConstructorExpr() final = default;

		private:
		std::vector<std::string> names;
	};

	class DirectArrayConstructorExpr: public ArrayConstructorExpr
	{
		public:
		DirectArrayConstructorExpr(SourceRange loc, vectorUnique<Expr> arguments)
				: ArrayConstructorExpr(
							loc,
							std::move(arguments),
							ExprKind::DirectArrayConstructorExpression)
		{
		}
		DirectArrayConstructorExpr(
				SourceRange loc, std::unique_ptr<ExprList> arguments)
				: ArrayConstructorExpr(
							loc,
							arguments->takeVector(),
							ExprKind::DirectArrayConstructorExpression)
		{
		}
		~DirectArrayConstructorExpr() final = default;

		static bool classof(const Expr* e)
		{
			return e->getKind() == ExprKind::DirectArrayConstructorExpression;
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
	};

	template<typename RappresentationType, BuiltinType T, Expr::ExprKind Kind>
	class LiteralExpr: public Expr
	{
		public:
		LiteralExpr(SourceRange loc, RappresentationType value)
				: Expr(loc, Type(T), Kind), value(std::move(value))
		{
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~LiteralExpr() final = default;

		static bool classof(const Expr* e) { return e->getKind() >= Kind; }
		[[nodiscard]] const RappresentationType& getValue() const { return value; }

		private:
		RappresentationType value;
	};

	class IfElseExpr: public ExprList
	{
		public:
		IfElseExpr(
				SourceRange loc,
				UniqueExpr ifCondition,
				UniqueExpr ifExpr,
				UniqueExpr elseExpr = nullptr)
				: ExprList(loc, Type(BuiltinType::Unknown), ExprKind::IfElseExpr)
		{
			getExpressions().emplace_back(std::move(ifCondition));
			getExpressions().emplace_back(std::move(ifExpr));
			if (elseExpr.get() != nullptr)
				getExpressions().emplace_back(std::move(elseExpr));
		}

		IfElseExpr(
				SourceRange loc,
				vectorUnique<Expr> ifThenExpressions,
				UniqueExpr elseExp = nullptr)
				: ExprList(
							loc,
							Type(BuiltinType::Unknown),
							ExprKind::IfElseExpr,
							std::move(ifThenExpressions))
		{
			if (elseExp.get() != nullptr)
				getExpressions().emplace_back(std::move(elseExp));
		}

		~IfElseExpr() final = default;

		static bool classof(const Expr* e)
		{
			return e->getKind() == ExprKind::IfElseExpr;
		}
		[[nodiscard]] const Expr* getFinalElse() const
		{
			if (!hasFinalElse())
				return nullptr;
			return getExpressions().back().get();
		}
		[[nodiscard]] bool hasFinalElse() const { return size() % 2 != 0; }

		[[nodiscard]] llvm::Error isConsistent() const;

		[[nodiscard]] const Expr* getCondition(int index) const
		{
			return at(index * 2);
		}
		[[nodiscard]] const Expr* getExpression(int index) const
		{
			return at(index * 2 + 1);
		}
	};

	class ArrayConcatExpr: public ExprList
	{
		public:
		ArrayConcatExpr(SourceRange loc, vectorUnique<ExprList> list)
				: ExprList(
							loc, Type(BuiltinType::Unknown), ExprKind::ArrayConcatExpression)
		{
			for (auto& expr : list)
				getExpressions().emplace_back(std::move(expr));
		}

		static bool classof(const Expr* e)
		{
			return e->getKind() == ExprKind::ArrayConcatExpression;
		}
		~ArrayConcatExpr() final = default;

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
	};

	using IntLiteralExpr =
			LiteralExpr<int, BuiltinType::Integer, Expr::ExprKind::IntLiteralExpr>;
	using BoolLiteralExpr =
			LiteralExpr<bool, BuiltinType::Boolean, Expr::ExprKind::BoolLiteralExpr>;
	using StringLiteralExpr = LiteralExpr<
			std::string,
			BuiltinType::String,
			Expr::ExprKind::StringLiteralExpr>;
	using FloatLiteralExpr =
			LiteralExpr<double, BuiltinType::Float, Expr::ExprKind::FloatLiteralExpr>;

}	// namespace modelica
