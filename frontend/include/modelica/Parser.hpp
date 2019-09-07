#pragma once
#include <memory>

#include "llvm/Support/Allocator.h"
#include "modelica/AST/Class.hpp"
#include "modelica/AST/Equation.hpp"
#include "modelica/AST/Expr.hpp"
#include "modelica/AST/Statement.hpp"
#include "modelica/Lexer.hpp"
#include "modelica/LexerStateMachine.hpp"
#include "modelica/ParserErrors.hpp"

namespace modelica
{
	/**
	 * Just an alias to avoid aving to write Expected<std::unique_ptr<...>> all
	 * the time
	 *
	 * The parser follow this idiom provided by the expected class
	 * auto var = parseSomething();
	 * if (!var) return var.takeError();
	 *
	 * This is used to try to perform an action, if it fails it returns the
	 * failure to the caller, else it returns the ast node.
	 */
	template<typename T>
	using ExpectedUnique = llvm::Expected<std::unique_ptr<T>>;

	/**
	 * The parser encapsulates the lexer but not he memory where string we are
	 * reading is held. It expones all the grammatical rules that are avialable
	 * in the grammar (can be found at page ~ 265 of the 3.4 doc).
	 *
	 */
	class Parser
	{
		public:
		Parser(const std::string& source)
				: lexer(source), current(lexer.scan()), undo(Token::End)
		{
		}

		Parser(const char* source)
				: lexer(source), current(lexer.scan()), undo(Token::End)
		{
		}

		/**
		 * This called every time a rule wishes to create a AST member,
		 * it is placed here because we may wish one day to move to
		 * version where we are controlling the memory with a allocator.
		 *
		 * After it has been created it will perform a check that the
		 * node is consistent.
		 */
		template<typename Type, typename... Args>
		[[nodiscard]] ExpectedUnique<Type> makeNode(
				const SourcePosition& initPoint, Args&&... args)
		{
			static_assert(
					std::is_base_of<Expr, Type>::value ||
							std::is_base_of<Equation, Type>::value ||
							std::is_base_of<Statement, Type>::value ||
							std::is_base_of<Declaration, Type>::value,
					"Type was not part of AST");
			auto ptr = std::make_unique<Type>(
					SourceRange(initPoint, getPosition()), std::forward<Args>(args)...);

			if (auto error = ptr->isConsistent())
				return std::move(error);
			return std::move(ptr);
		}

		[[nodiscard]] ExpectedUnique<Expr> primary();
		[[nodiscard]] ExpectedUnique<Expr> expression();
		[[nodiscard]] ExpectedUnique<Expr> simpleExpression();
		[[nodiscard]] ExpectedUnique<Expr> logicalExpression();
		[[nodiscard]] ExpectedUnique<Expr> logicalTerm();
		[[nodiscard]] ExpectedUnique<Expr> logicalFactor();
		[[nodiscard]] ExpectedUnique<Expr> relation();
		[[nodiscard]] std::optional<BinaryExprOp> relationalOperator();
		[[nodiscard]] std::optional<BinaryExprOp> maybeAddOperator();
		[[nodiscard]] std::optional<BinaryExprOp> maybeMulOperator();
		[[nodiscard]] std::optional<UnaryExprOp> maybeUnaryAddOperator();
		[[nodiscard]] ExpectedUnique<Expr> arithmeticExpression();
		[[nodiscard]] ExpectedUnique<Expr> term();
		[[nodiscard]] ExpectedUnique<Expr> factor();
		[[nodiscard]] ExpectedUnique<Expr> componentReference();
		[[nodiscard]] ExpectedUnique<NamedArgumentExpr> namedArgument();
		[[nodiscard]] llvm::Expected<vectorUnique<Expr>> namedArguments();
		[[nodiscard]] llvm::Expected<vectorUnique<Expr>> functionArguments();
		[[nodiscard]] llvm::Expected<vectorUnique<Expr>>
		functionArgumentsNonFirst();
		[[nodiscard]] ExpectedUnique<ExprList> expressionList();
		[[nodiscard]] ExpectedUnique<ArrayConstructorExpr> arrayArguments();
		[[nodiscard]] llvm::Expected<vectorUnique<Expr>> arraySubscript();
		[[nodiscard]] ExpectedUnique<Expr> subScript();
		[[nodiscard]] llvm::Expected<std::vector<std::string>> name();
		[[nodiscard]] ExpectedUnique<Expr> partialCall();
		[[nodiscard]] ExpectedUnique<Expr> functionArgument();
		[[nodiscard]] ExpectedUnique<Equation> equation();
		[[nodiscard]] ExpectedUnique<Equation> ifEquation();
		[[nodiscard]] ExpectedUnique<Equation> forEquation();
		[[nodiscard]] ExpectedUnique<Equation> whenEquation();
		[[nodiscard]] llvm::Expected<vectorUnique<Equation>> equationList(
				const std::vector<Token>& stopTokens);
		[[nodiscard]] llvm::Expected<std::pair<UniqueEq, UniqueExpr>> ifBrach(
				const std::vector<Token>& stopTokens);
		[[nodiscard]] llvm::Expected<std::pair<std::string, UniqueExpr>> forIndex();

		[[nodiscard]] ExpectedUnique<ClassDecl> classDefinition();
		[[nodiscard]] ExpectedUnique<ClassDecl> classSpecifier();
		[[nodiscard]] ExpectedUnique<ClassDecl> selectClassSpecifier();
		[[nodiscard]] ExpectedUnique<ClassDecl> longClassSpecifier();
		[[nodiscard]] ExpectedUnique<ImportClause> importClause();
		[[nodiscard]] llvm::Expected<std::string> stringComment();
		[[nodiscard]] llvm::Expected<TypeSpecifier> typeSpecifier();
		[[nodiscard]] llvm::Expected<DeclarationName> declaration();

		/**
		 * First bool of tuple is partiality,
		 * second bool is purity
		 */
		[[nodiscard]] llvm::Expected<std::tuple<bool, bool, ClassDecl::SubType>>
		classPrefixes();

		[[nodiscard]] ExpectedUnique<Statement> whenStatement();
		[[nodiscard]] ExpectedUnique<Statement> forStatement();
		[[nodiscard]] ExpectedUnique<Statement> ifStatement();
		[[nodiscard]] ExpectedUnique<Statement> whileStatement();
		[[nodiscard]] ExpectedUnique<Statement> statement();
		[[nodiscard]] ExpectedUnique<Declaration> componentClause1();
		[[nodiscard]] ExpectedUnique<Declaration> componentClause();
		[[nodiscard]] ExpectedUnique<Declaration> componentDeclaration1();
		[[nodiscard]] ExpectedUnique<Declaration> componentDeclaration();
		[[nodiscard]] ExpectedUnique<Declaration> conditionAttribute();
		[[nodiscard]] llvm::Expected<vectorUnique<Declaration>> componentList();
		[[nodiscard]] ExpectedUnique<Declaration> comment();
		[[nodiscard]] ExpectedUnique<Declaration> classModification();
		[[nodiscard]] ExpectedUnique<Declaration> modification();
		[[nodiscard]] llvm::Expected<ComponentClause::Prefix> typePrefix();
		[[nodiscard]] ExpectedUnique<Declaration> elementReplaceable(
				bool each, bool fnl);
		[[nodiscard]] ExpectedUnique<Declaration> constrainingClause();
		[[nodiscard]] ExpectedUnique<Declaration> extendClause();
		[[nodiscard]] ExpectedUnique<Declaration> annotation();

		[[nodiscard]] ExpectedUnique<Declaration> elementRedeclaration();
		[[nodiscard]] ExpectedUnique<Declaration> argument();
		[[nodiscard]] ExpectedUnique<Declaration> elementModification(
				bool each, bool fnl);

		[[nodiscard]] llvm::Expected<vectorUnique<Statement>> statementList(
				const std::vector<Token>& stopTokens);
		[[nodiscard]] llvm::Expected<std::pair<UniqueStmt, UniqueExpr>> ifStmtBrach(
				const std::vector<Token>& stopTokes);

		template<Token token, typename T>
		[[nodiscard]] std::optional<ExpectedUnique<Expr>> functionCall()
		{
			SourcePosition currentPos = getPosition();
			if (accept<token>())
			{
				if (!expect(Token::LPar))
					return llvm::make_error<UnexpectedToken>(current);

				if (accept<Token::RPar>())
					return makeNode<T>(currentPos, vectorUnique<Expr>());

				auto arguments = functionArguments();
				if (!arguments)
					return arguments.takeError();

				if (!expect(Token::RPar))
					return llvm::make_error<UnexpectedToken>(current);

				return makeNode<T>(currentPos, move(*arguments));
			}
			return std::nullopt;
		}

		/**
		 * Return the current position in the source stream
		 */
		[[nodiscard]] SourcePosition getPosition() const
		{
			return SourcePosition(lexer.getCurrentLine(), lexer.getCurrentColumn());
		}

		private:
		/**
		 * regular accept, if the current token it then the next one will be read
		 * and true will be returned, else false.
		 */
		bool accept(Token t)
		{
			if (current == t)
			{
				next();
				return true;
			}
			return false;
		}

		/**
		 * fancy overloads if you know at compile time
		 * which token you want.
		 */
		template<Token t>
		bool accept()
		{
			if (current == t)
			{
				next();
				return true;
			}
			return false;
		}

		/**
		 * return a error if was token was not accepted.
		 * Notice that since errors are returned instead of
		 * being thrown this mean that there is no really difference
		 * between accept and expect. It is used here to signal that if
		 * an expect fail then the function will terminate immediatly,
		 * a accept is allowed to continue instead so that
		 * it is less confusing to people that are used to the accept expect
		 * notation.
		 *
		 * expect returns an Expected bool instead of a llvm::Error
		 * beacause to check for errors in a expected you do if (!expected)
		 * and in a llvm::Error you do if (error), this would be so confusing
		 * that this decision was better.
		 */
		llvm::Expected<bool> expect(Token t);

		/**
		 * reads the next token
		 */
		void next()
		{
			if (undo != Token::End)
			{
				current = undo;
				undo = Token::End;
				return;
			}

			current = lexer.scan();
		}

		/**
		 * Unfortunately the grammar in the
		 * spec is written in such a way that
		 * there is a single point where you need a two lookhaed
		 * to tell the difference between a component reference and a
		 * named argument. Istead of adding a real two lookhaed
		 * or to compleatly change factor the grammar we can provide the ability
		 * to undo the last accept. it is used only to tell apart that particular
		 * case.
		 */
		void undoScan(Token t)
		{
			undo = current;
			current = t;
		}
		Lexer<ModelicaStateMachine> lexer;
		Token current;
		Token undo;
	};

}	// namespace modelica
