#pragma once

#include "modelica/AST/Class.hpp"
#include "modelica/AST/Equation.hpp"
#include "modelica/AST/Expr.hpp"
#include "modelica/AST/Statement.hpp"

namespace modelica
{
	/**
	 * This is just handy class to extend if you wish
	 * to write bottom up visitors, just publicly extend this and
	 * provide the overloads to the AST types you care about.
	 *
	 * Return nullptr in a function to delete the node you received.
	 * Return the node itself to keep it.
	 * Return another node to change it.
	 */
	struct BaseVisitor
	{
		public:
		template<typename T>
		std::unique_ptr<T> visit(std::unique_ptr<T> ptr)
		{
			return ptr;
		}
	};

	/**
	 * Direction is the type trait that contains the informatiosn regarding
	 * how to explore the tree. This file provide a bottomUp direction
	 * and a topDown direction.
	 *
	 * ASTNode is the node that is being being elaborated, and visitor is
	 * the user provided class that will recive one at the time each node.
	 *
	 * This is the final overload for select descendant where the list of
	 * possible types that the current variable had was exausted.
	 * Since it was exausted then the node must be a ASTNode and not a descendant
	 * and it is passed to the visitor.
	 */
	template<typename Direction, typename ASTNode, typename Visitor>
	auto selectDescendant(std::unique_ptr<ASTNode> node, Visitor& visitor)
			-> decltype(visitor.visit(std::move(node)))
	{
		return visitor.visit(std::move(node));
	}

	/**
	 * This is the overload in which we do not have exausted the list of
	 * types that a node might have. what we do is we take the first
	 * element of the parameter pack, we check if the node is of that type,
	 * if it is we pass it to the direction type trait to let him decide what
	 * to do with it, otherwise we discard CurrentDescant and we recurre until no
	 * option was left, at that point the other overload is called.
	 */
	template<
			typename Direction,
			typename ASTNode,
			typename Visitor,
			typename CurrentDescendant,
			typename... Descendents>
	auto selectDescendant(std::unique_ptr<ASTNode> node, Visitor& visitor)
			-> decltype(visitor.visit(std::move(node)))
	{
		if (node == nullptr)
			return node;
		if (llvm::isa<CurrentDescendant>(node))
		{
			std::unique_ptr<CurrentDescendant> casted =
					llvm::cast<CurrentDescendant>(std::move(node));
			return Direction::visit(std::move(casted), visitor);
		}
		return selectDescendant<Direction, ASTNode, Visitor, Descendents...>(
				std::move(node), visitor);
	}

	/**
	 * This is the list of direct descendant of an expression provided as an
	 * overload of selectDescendant, a Direction typetrait can use this function
	 * to resolve the kind of a node and invoke the appropriate method inside the
	 * Direction typetrait itself.
	 */
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

	/**
	 * This is the list of direct descendant of an equation provided as an
	 * overload of selectDescendant, a Direction typetrait can use this function
	 * to resolve the kind of a node and invoke the appropriate method inside the
	 * Direction typetrait itself.
	 */
	template<typename Direction, typename Visitor>
	const auto EqSelectDescendant = selectDescendant<
			Direction,
			Equation,
			Visitor,
			SimpleEquation,
			ConnectClause,
			CallEquation,
			CompositeEquation>;

	/**
	 *
	 * This is the list of direct descendant of an statement provided as an
	 * overload of selectDescendant, a Direction typetrait can use this function
	 * to resolve the kind of a node and invoke the appropriate method inside the
	 * Direction typetrait itself.
	 *
	 */
	template<typename Direction, typename Visitor>
	const auto StatementSelectDescendant = selectDescendant<
			Direction,
			Statement,
			Visitor,
			CompositeStatement,
			CallStatement,
			AssignStatement,
			BreakStatement,
			ReturnStatement>;

	/**
	 *
	 * This is the list of direct descendant of a composite statement provided as
	 * an overload of selectDescendant, a Direction typetrait can use this
	 * function to resolve the kind of a node and invoke the appropriate method
	 * inside the Direction typetrait itself.
	 *
	 */
	template<typename Direction, typename Visitor>
	const auto CompositeStatementSelectDescendant = selectDescendant<
			Direction,
			CompositeStatement,
			Visitor,
			IfStatement,
			WhenStatement,
			WhileStatement,
			ForStatement>;

	/**
	 * This is the list of direct descendant of an composite provided
	 * as an overload of selectDescendant, a Direction typetrait can
	 * use this function to resolve the kind of a node and invoke the
	 * appropriate method inside the Direction typetrait itself.
	 */
	template<typename Direction, typename Visitor>
	const auto CompositeEqDescendant = selectDescendant<
			Direction,
			CompositeEquation,
			Visitor,
			IfEquation,
			ForEquation,
			WhenEquation>;

	/**
	 * This is the list of direct descendant of an FunctionExpr provided as an
	 * overload of selectDescendant, a Direction typetrait can use this function
	 * to resolve the kind of a node and invoke the appropriate method inside the
	 * Direction typetrait itself.
	 */
	template<typename Direction, typename Visitor>
	const auto FunctionEqDescendant = selectDescendant<
			Direction,
			FunctionCallExpr,
			Visitor,
			DerFunctionCallExpr,
			InitialFunctionCallExpr,
			PureFunctionCallExpr,
			ComponentFunctionCallExpr>;

	/**
	 * This is the list of direct descendant of an arrayConstructorExpr provided
	 * as an overload of selectDescendant, a Direction typetrait can use this
	 * function to resolve the kind of a node and invoke the appropriate method
	 * inside the Direction typetrait itself.
	 */
	template<typename Direction, typename Visitor>
	const auto ArrayConstructorExprDescendant = selectDescendant<
			Direction,
			ArrayConstructorExpr,
			Visitor,
			DirectArrayConstructorExpr,
			ForInArrayConstructorExpr>;

	/**
	 * This is the list of direct descendant of an ExprList provided as an
	 * overload of selectDescendant, a Direction typetrait can use this function
	 * to resolve the kind of a node and invoke the appropriate method inside the
	 * Direction typetrait itself.
	 */
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
			ComponentReferenceExpr,
			ArraySubscriptionExpr>;

	/**
	 * This is the list of direct descendant of an decl provided as an
	 * overload of selectDescendant, a Direction typetrait can use this function
	 * to resolve the kind of a node and invoke the appropriate method inside the
	 * Direction typetrait itself.
	 */
	template<typename Direction, typename Visitor>
	const auto DeclDescendant = selectDescendant<
			Direction,
			Declaration,
			Visitor,
			CompositeDecl,
			ExprCompositeDecl,
			EqCompositeDecl,
			StatementCompositeDecl>;

	/**
	 * This is the list of direct descendant of an compositeDecl provided as an
	 * overload of selectDescendant, a Direction typetrait can use this function
	 * to resolve the kind of a node and invoke the appropriate method inside the
	 * Direction typetrait itself.
	 */
	template<typename Direction, typename Visitor>
	const auto CompositeDeclDescendant = selectDescendant<
			Direction,
			CompositeDecl,
			Visitor,
			ClassModification,
			Composition,
			ExtendClause,
			ElementList,
			EnumerationLiteral,
			Element,
			ClassDecl,
			CompositionSection,
			OverridingClassModification,
			ImportClause,
			ComponentDeclaration,
			Annotation,
			ComponentClause,
			Redeclaration,
			ReplecableModification,
			ConstrainingClause,
			ElementModification>;

	/**
	 * This is the list of direct descendant of an ClassDecl provided as an
	 * overload of selectDescendant, a Direction typetrait can use this function
	 * to resolve the kind of a node and invoke the appropriate method inside the
	 * Direction typetrait itself.
	 */
	template<typename Direction, typename Visitor>
	const auto ClassDeclDescendant = selectDescendant<
			Direction,
			ClassDecl,
			Visitor,
			EnumerationClass,
			LongClassDecl,
			ShortClassDecl,
			DerClassDecl>;

	/**
	 * This is the list of direct descendant of an ExprCompositeDecl provided as
	 * an overload of selectDescendant, a Direction typetrait can use this
	 * function to resolve the kind of a node and invoke the appropriate method
	 * inside the Direction typetrait itself.
	 */
	template<typename Direction, typename Visitor>
	const auto ExprCompositeDeclDescendant = selectDescendant<
			Direction,
			ExprCompositeDecl,
			Visitor,
			ConditionAttribute,
			ExternalFunctionCall,
			SimpleModification,
			ArraySubscriptionDecl>;

	/**
	 * This is the list of direct descendant of an EqCompositeDecl provided as
	 * an overload of selectDescendant, a Direction typetrait can use this
	 * function to resolve the kind of a node and invoke the appropriate method
	 * inside the Direction typetrait itself.
	 */
	template<typename Direction, typename Visitor>
	const auto EqCompositeDeclDescendant =
			selectDescendant<Direction, EqCompositeDecl, Visitor, EquationSection>;

	/**
	 * This is the list of direct descendant of an StatementCompositeDecl provided
	 * as an overload of selectDescendant, a Direction typetrait can use this
	 * function to resolve the kind of a node and invoke the appropriate method
	 * inside the Direction typetrait itself.
	 */
	template<typename Direction, typename Visitor>
	const auto StatementCompositeDeclDescendant = selectDescendant<
			Direction,
			StatementCompositeDecl,
			Visitor,
			AlgorithmSection>;

	/**
	 * This is the typetrait that is invoked when trying to visit a AST
	 */
	class TopDownDirection
	{
		public:
		/**
		 * this is the default beahviour when the node type does not have any
		 * possible descendant. If no better matching overload is found this will be
		 * selected and it will send the node to the visitor, calling the
		 * appropriate overload there.
		 */
		template<typename ASTNode, typename Visitor>
		static auto visit(std::unique_ptr<ASTNode> node, Visitor& visitor)
		{
			return visitor.visit(std::move(node));
		}

		/**
		 * This is a overload for a node that does not introduce subobjects, but his
		 * type is extended by other types.  When that is the case we just call the
		 * alias to SelectDescent that already include all the possible extended
		 * types and we let him resolve that.
		 *
		 */
		template<typename Visitor>
		static auto visit(std::unique_ptr<Expr> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			return ExprSelectDescendant<TopDownDirection, Visitor>(
					std::move(node), visitor);
		}

		/**
		 * This is an overload for a node that introduces a list of children that
		 * need to be navigated as well. Since it is top down first we save a
		 * pointer to the object, since the visitor might change it. Then we use the
		 * selectDescendant alias to send it to the correct overload of the visitor.
		 * Finaly if the node has not been changed we visit every children, then
		 * after children visit is called to allow the visitor to keep track of the
		 * depth if he wishes to do so
		 */
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

			visitor.afterChildrenVisit(old);

			return toReturn;
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<Statement> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			auto old = node.get();
			auto toReturn = StatementSelectDescendant<TopDownDirection, Visitor>(
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

			visitor.afterChildrenVisit(old);

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
		static auto visit(
				std::unique_ptr<CompositeStatement> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			auto old = node.get();
			auto toReturn =
					CompositeStatementSelectDescendant<TopDownDirection, Visitor>(
							std::move(node), visitor);

			if (old != toReturn.get())
				return std::move(toReturn);

			auto casted = llvm::cast<CompositeStatement>(move(toReturn));
			std::transform(
					casted->stmtBegin(),
					casted->stmtEnd(),
					casted->stmtBegin(),
					[&visitor](std::unique_ptr<Statement>& node) {
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
			visitor.afterChildrenVisit(casted.get());

			return casted;
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<Declaration> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			return DeclDescendant<TopDownDirection, Visitor>(
					std::move(node), visitor);
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<ClassDecl> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			return ClassDeclDescendant<TopDownDirection, Visitor>(
					std::move(node), visitor);
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<CompositeDecl> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			auto old = node.get();
			auto toReturn = CompositeDeclDescendant<TopDownDirection, Visitor>(
					std::move(node), visitor);

			if (old != toReturn.get())
				return std::move(toReturn);

			auto casted = llvm::cast<CompositeDecl>(move(toReturn));
			std::transform(
					casted->begin(),
					casted->end(),
					casted->begin(),
					[&visitor](std::unique_ptr<Declaration>& node) {
						return visit(std::move(node), visitor);
					});
			casted->removeNulls();
			visitor.afterChildrenVisit(casted.get());

			return casted;
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<EqCompositeDecl> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			auto old = node.get();
			auto toReturn = EqCompositeDeclDescendant<TopDownDirection, Visitor>(
					std::move(node), visitor);

			if (old != toReturn.get())
				return std::move(toReturn);

			auto casted = llvm::cast<EqCompositeDecl>(move(toReturn));
			std::transform(
					casted->begin(),
					casted->end(),
					casted->begin(),
					[&visitor](std::unique_ptr<Equation>& node) {
						return visit(std::move(node), visitor);
					});
			casted->removeNulls();
			visitor.afterChildrenVisit(casted.get());

			return casted;
		}

		template<typename Visitor>
		static auto visit(
				std::unique_ptr<StatementCompositeDecl> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			auto old = node.get();
			auto toReturn =
					StatementCompositeDeclDescendant<TopDownDirection, Visitor>(
							std::move(node), visitor);

			if (old != toReturn.get())
				return std::move(toReturn);

			auto casted = llvm::cast<StatementCompositeDecl>(move(toReturn));
			std::transform(
					casted->begin(),
					casted->end(),
					casted->begin(),
					[&visitor](std::unique_ptr<Statement>& node) {
						return visit(std::move(node), visitor);
					});
			casted->removeNulls();
			visitor.afterChildrenVisit(casted.get());

			return casted;
		}
		template<typename Visitor>
		static auto visit(std::unique_ptr<ExprCompositeDecl> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			auto old = node.get();
			auto toReturn = ExprCompositeDeclDescendant<TopDownDirection, Visitor>(
					std::move(node), visitor);

			if (old != toReturn.get())
				return std::move(toReturn);

			auto casted = llvm::cast<ExprCompositeDecl>(move(toReturn));
			std::transform(
					casted->begin(),
					casted->end(),
					casted->begin(),
					[&visitor](std::unique_ptr<Expr>& node) {
						return visit(std::move(node), visitor);
					});
			casted->removeNulls();
			visitor.afterChildrenVisit(casted.get());

			return casted;
		}
	};

	/**
	 * The bottom up direction is almost exactly like the top down direction,
	 * expect it is easier to to handle objects that get changed by the visitor
	 */
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

		/**
		 * Since the children are trasnformed before the parent we don't care
		 * about checking if the parent is the same as we did in the top down
		 * version.
		 */
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
		static auto visit(std::unique_ptr<Statement> node, Visitor& visitor)
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

			return StatementSelectDescendant<BottomUpDirection, Visitor>(
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
		static auto visit(
				std::unique_ptr<CompositeStatement> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			std::transform(
					node->stmtBegin(),
					node->stmtEnd(),
					node->stmtBegin(),
					[&visitor](std::unique_ptr<Equation>& node) {
						return visit(std::move(node), visitor);
					});
			node->removeNullEq();

			return CompositeStatementSelectDescendant<BottomUpDirection, Visitor>(
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

		template<typename Visitor>
		static auto visit(std::unique_ptr<Declaration> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			return DeclDescendant<BottomUpDirection, Visitor>(
					std::move(node), visitor);
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<ClassDecl> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			return ClassDeclDescendant<BottomUpDirection, Visitor>(
					std::move(node), visitor);
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<CompositeDecl> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			std::transform(
					node->begin(),
					node->end(),
					node->begin(),
					[&visitor](std::unique_ptr<Declaration>& node) {
						return visit(std::move(node), visitor);
					});
			node->removeNulls();

			return CompositeDeclDescendant<BottomUpDirection, Visitor>(
					std::move(node), visitor);
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<EqCompositeDecl> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			std::transform(
					node->begin(),
					node->end(),
					node->begin(),
					[&visitor](std::unique_ptr<Equation>& node) {
						return visit(std::move(node), visitor);
					});
			node->removeNulls();

			return EqCompositeDeclDescendant<BottomUpDirection, Visitor>(
					std::move(node), visitor);
		}

		template<typename Visitor>
		static auto visit(
				std::unique_ptr<StatementCompositeDecl> node, Visitor& visitor)
				-> decltype(visitor.visit(std::move(node)))
		{
			std::transform(
					node->begin(),
					node->end(),
					node->begin(),
					[&visitor](std::unique_ptr<Statement>& node) {
						return visit(std::move(node), visitor);
					});
			node->removeNulls();

			return EqCompositeDeclDescendant<BottomUpDirection, Visitor>(
					std::move(node), visitor);
		}

		template<typename Visitor>
		static auto visit(std::unique_ptr<ExprCompositeDecl> node, Visitor& visitor)
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

			return ExprCompositeDeclDescendant<BottomUpDirection, Visitor>(
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
	std::unique_ptr<Statement> topDownVisit(
			std::unique_ptr<Statement> expr, Visitor& visitor)
	{
		return TopDownDirection::visit(std::move(expr), visitor);
	}

	template<typename Visitor>
	std::unique_ptr<Declaration> topDownVisit(
			std::unique_ptr<Declaration> expr, Visitor& visitor)
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
	template<typename Visitor>
	std::unique_ptr<Statement> bottomUpVisit(
			std::unique_ptr<Statement> expr, Visitor& visitor)
	{
		return BottomUpDirection::visit(std::move(expr), visitor);
	}
	template<typename Visitor>
	std::unique_ptr<Declaration> bottomUpVisit(
			std::unique_ptr<Declaration> expr, Visitor& visitor)
	{
		return BottomUpDirection::visit(std::move(expr), visitor);
	}
}	// namespace modelica
