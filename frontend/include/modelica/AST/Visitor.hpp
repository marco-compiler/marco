#pragma once

#include "modelica/AST/Equation.hpp"
#include "modelica/AST/Expr.hpp"

namespace modelica
{
	struct BaseVisitor
	{
		template<typename T>
		std::unique_ptr<T> visit(std::unique_ptr<T> ptr)
		{
			return ptr;
		}
	};

	template<typename Direction, typename ASTNode, typename Visitor>
	auto visit(std::unique_ptr<ASTNode> node, Visitor& visitor);

	template<typename Direction, typename ASTNode, typename Visitor>
	auto selectDescendant(std::unique_ptr<ASTNode> node, Visitor& visitor)
			-> decltype(visitor.visit(std::move(node)))
	{
		return visitor.visit(std::move(node));
	}

	template<
			typename Direction,
			typename ASTNode,
			typename Visitor,
			typename CurrentDescendant,
			typename... Descendents>
	auto selectDescendant(std::unique_ptr<ASTNode> node, Visitor& visitor)
			-> decltype(visitor.visit(std::move(node)))
	{
		if (llvm::isa<CurrentDescendant>(node))
		{
			std::unique_ptr<CurrentDescendant> casted =
					llvm::cast<CurrentDescendant>(std::move(node));
			return Direction::visit(std::move(casted), visitor);
		}
		return selectDescendant<Direction, ASTNode, Visitor, Descendents...>(
				std::move(node), visitor);
	}

	template<typename Direction, typename Visitor>
	const auto ExprSelectDescendant = selectDescendant<
			Direction,
			Expr,
			Visitor,
			BoolLiteralExpr,
			ExprList,
			IntLiteralExpr,
			EndExpr,
			StringLiteralExpr,
			FloatLiteralExpr,
			AcceptAllExpr>;

	template<typename Direction, typename Visitor>
	const auto EqSelectDescendant = selectDescendant<
			Direction,
			Equation,
			Visitor,
			SimpleEquation,
			ConnectClause,
			CallEquation,
			CompositeEquation>;

	template<typename Direction, typename Visitor>
	const auto CompositeEqDescendant = selectDescendant<
			Direction,
			CompositeEquation,
			Visitor,
			IfEquation,
			ForEquation,
			WhenEquation>;

	template<typename Direction, typename Visitor>
	const auto FunctionEqDescendant = selectDescendant<
			Direction,
			FunctionCallExpr,
			Visitor,
			DerFunctionCallExpr,
			InitialFunctionCallExpr,
			PureFunctionCallExpr,
			ComponentFunctionCallExpr>;

	template<typename Direction, typename Visitor>
	const auto ArrayConstructorExprDescendant = selectDescendant<
			Direction,
			ArrayConstructorExpr,
			Visitor,
			DirectArrayConstructorExpr,
			ForInArrayConstructorExpr>;

	template<typename Direction, typename Visitor>
	const auto ExprListDescendant = selectDescendant<
			Direction,
			ExprList,
			Visitor,
			NamedArgumentExpr,
			IfElseExpr,
			BinaryExpr,
			UnaryExpr,
			IfElseExpr,
			ArrayConcatExpr,
			RangeExpr,
			ArrayConstructorExpr,
			FunctionCallExpr,
			ComponentReferenceExpr>;

	class TopDownDirection
	{
		public:
		template<typename ASTNode, typename Visitor>
		static auto visit(std::unique_ptr<ASTNode> node, Visitor& visitor)
		{
			return visitor.visit(std::move(node));
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<Expr> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			return ExprSelectDescendant<TopDownDirection, Visitor>(
					std::move(node), visitor);
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<Equation> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			auto old = node.get();
			auto toReturn = EqSelectDescendant<TopDownDirection, Visitor>(
					std::move(node), visitor);

			if (old != toReturn.get())
				return std::move(toReturn);

			std::transform(
					old->exprBegin(),
					old->exprEnd(),
					old->exprBegin(),
					[&visitor](std::unique_ptr<Expr>& node) {
						return visit(std::move(node), visitor);
					});
			old->removeNullExpr();

			return toReturn;
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<CompositeEquation> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			auto old = node.get();
			auto toReturn = CompositeEqDescendant<TopDownDirection, Visitor>(
					std::move(node), visitor);

			if (old != toReturn.get())
				return std::move(toReturn);

			auto casted = llvm::cast<CompositeEquation>(move(toReturn));
			std::transform(
					casted->eqBegin(),
					casted->eqEnd(),
					casted->eqBegin(),
					[&visitor](std::unique_ptr<Equation>& node) {
						return visit(std::move(node), visitor);
					});
			casted->removeNullEq();

			return casted;
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<FunctionCallExpr> node, Visitor& visitor)
		{
			return FunctionEqDescendant<TopDownDirection, Visitor>(
					std::move(node), visitor);
		}

		template<typename Visitor>
		static auto visit(
				std::unique_ptr<ArrayConstructorExpr> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			return ArrayConstructorExprDescendant<TopDownDirection, Visitor>(
					std::move(node), visitor);
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<ExprList> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			auto old = node.get();
			auto toReturn = ExprListDescendant<TopDownDirection, Visitor>(
					std::move(node), visitor);

			if (old != toReturn.get())
				return std::move(toReturn);

			auto casted = llvm::cast<ExprList>(move(toReturn));
			std::transform(
					casted->begin(),
					casted->end(),
					casted->begin(),
					[&visitor](std::unique_ptr<Expr>& node) {
						return visit(std::move(node), visitor);
					});
			casted->removeNulls();

			return casted;
		}
	};

	class BottomUpDirection
	{
		public:
		template<typename ASTNode, typename Visitor>
		static auto visit(std::unique_ptr<ASTNode> node, Visitor& visitor)
		{
			return visitor.visit(std::move(node));
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<Expr> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			return ExprSelectDescendant<BottomUpDirection, Visitor>(
					std::move(node), visitor);
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<Equation> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			std::transform(
					node->exprBegin(),
					node->exprEnd(),
					node->exprBegin(),
					[&visitor](std::unique_ptr<Expr>& node) {
						return visit(std::move(node), visitor);
					});
			node->removeNullExpr();

			return EqSelectDescendant<BottomUpDirection, Visitor>(
					std::move(node), visitor);
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<CompositeEquation> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			std::transform(
					node->eqBegin(),
					node->eqEnd(),
					node->eqBegin(),
					[&visitor](std::unique_ptr<Equation>& node) {
						return visit(std::move(node), visitor);
					});
			node->removeNullEq();

			return CompositeEqDescendant<BottomUpDirection, Visitor>(
					std::move(node), visitor);
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<FunctionCallExpr> node, Visitor& visitor)
		{
			return FunctionEqDescendant<BottomUpDirection, Visitor>(
					std::move(node), visitor);
		}

		template<typename Visitor>
		static auto visit(
				std::unique_ptr<ArrayConstructorExpr> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			return ArrayConstructorExprDescendant<BottomUpDirection, Visitor>(
					std::move(node), visitor);
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<ExprList> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			std::transform(
					node->begin(),
					node->end(),
					node->begin(),
					[&visitor](std::unique_ptr<Expr>& node) {
						return visit(std::move(node), visitor);
					});
			node->removeNulls();

			return ExprListDescendant<BottomUpDirection, Visitor>(
					std::move(node), visitor);
		}
	};

	template<typename Visitor>
	std::unique_ptr<Expr> topDownVisit(
			std::unique_ptr<Expr> expr, Visitor& visitor)
	{
		return TopDownDirection::visit(std::move(expr), visitor);
	}
	template<typename Visitor>
	std::unique_ptr<Equation> topDownVisit(
			std::unique_ptr<Equation> expr, Visitor& visitor)
	{
		return TopDownDirection::visit(std::move(expr), visitor);
	}

	template<typename Visitor>
	std::unique_ptr<Expr> bottomUpVisit(
			std::unique_ptr<Expr> expr, Visitor& visitor)
	{
		return BottomUpDirection::visit(std::move(expr), visitor);
	}
	template<typename Visitor>
	std::unique_ptr<Equation> bottomUpVisit(
			std::unique_ptr<Equation> expr, Visitor& visitor)
	{
		return BottomUpDirection::visit(std::move(expr), visitor);
	}

}	// namespace modelica
