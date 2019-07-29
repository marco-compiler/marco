#pragma once

#include "modelica/AST/Equation.hpp"
#include "modelica/AST/Expr.hpp"

namespace modelica
{
	struct Visitor
	{
		template<typename T>
		std::unique_ptr<T> visit(std::unique_ptr<T> ptr)
		{
			return ptr;
		}
	};

	template<typename ASTNode, typename Visitor>
	auto visit(std::unique_ptr<ASTNode> node, Visitor& visitor);

	template<typename ASTNode, typename Visitor>
	auto selectDescendant(std::unique_ptr<ASTNode> node, Visitor& visitor)
			-> decltype(visitor.visit(std::move(node)))
	{
		return visitor.visit(std::move(node));
	}

	template<
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
			return visit(std::move(casted), visitor);
		}
		return selectDescendant<ASTNode, Visitor, Descendents...>(
				std::move(node), visitor);
	}

	template<typename ASTNode, typename Visitor>
	auto visit(std::unique_ptr<ASTNode> node, Visitor& visitor)
	{
		return visitor.visit(std::move(node));
	}

	template<typename Visitor>
	auto visit(std::unique_ptr<Expr> node, Visitor& visitor)
			-> decltype(visitor.visit(std::move(node)))
	{
		return selectDescendant<
				Expr,
				Visitor,
				BoolLiteralExpr,
				ExprList,
				IntLiteralExpr,
				EndExpr,
				StringLiteralExpr,
				FloatLiteralExpr,
				AcceptAllExpr>(std::move(node), visitor);
	}

	template<typename Visitor>
	auto visit(std::unique_ptr<Equation> node, Visitor& visitor)
			-> decltype(visitor.visit(std::move(node)))
	{
		auto old = node.get();
		auto toReturn = selectDescendant<
				Equation,
				Visitor,
				SimpleEquation,
				ConnectClause,
				CallEquation,
				CompositeEquation>(std::move(node), visitor);

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
	auto visit(std::unique_ptr<CompositeEquation> node, Visitor& visitor)
			-> decltype(visitor.visit(std::move(node)))
	{
		auto old = node.get();
		auto toReturn = selectDescendant<
				CompositeEquation,
				Visitor,
				IfEquation,
				ForEquation,
				WhenEquation>(std::move(node), visitor);

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
	auto visit(std::unique_ptr<FunctionCallExpr> node, Visitor& visitor)
	{
		return selectDescendant<
				FunctionCallExpr,
				Visitor,
				DerFunctionCallExpr,
				InitialFunctionCallExpr,
				PureFunctionCallExpr,
				ComponentFunctionCallExpr>(std::move(node), visitor);
	}

	template<typename Visitor>
	auto visit(std::unique_ptr<ArrayConstructorExpr> node, Visitor& visitor)
			-> decltype(visitor.visit(std::move(node)))
	{
		return selectDescendant<
				ArrayConstructorExpr,
				Visitor,
				DirectArrayConstructorExpr,
				ForInArrayConstructorExpr>(std::move(node), visitor);
	}

	template<typename Visitor>
	auto visit(std::unique_ptr<ExprList> node, Visitor& visitor)
			-> decltype(visitor.visit(std::move(node)))
	{
		auto old = node.get();
		auto toReturn = selectDescendant<
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
				ComponentReferenceExpr>(std::move(node), visitor);

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

	template<typename Visitor>
	std::unique_ptr<Expr> topDownVisit(
			std::unique_ptr<Expr> expr, Visitor& visitor)
	{
		return visit(std::move(expr), visitor);
	}
	template<typename Visitor>
	std::unique_ptr<Equation> topDownVisit(
			std::unique_ptr<Equation> expr, Visitor& visitor)
	{
		return visit(std::move(expr), visitor);
	}

}	// namespace modelica
