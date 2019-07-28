#pragma once
#include <memory>

#include "llvm/Support/Allocator.h"
#include "modelica/AST/Equation.hpp"
#include "modelica/AST/Expr.hpp"
#include "modelica/Lexer.hpp"
#include "modelica/LexerStateMachine.hpp"
#include "modelica/ParserErrors.hpp"

namespace modelica
{
	template<typename T>
	using ExpectedUnique = llvm::Expected<std::unique_ptr<T>>;
	class Parser
	{
		public:
		Parser(const std::string& source): lexer(source), current(lexer.scan()) {}

		Parser(const char* source): lexer(source), current(lexer.scan()) {}

		template<typename Type, typename... Args>
		[[nodiscard]] ExpectedUnique<Type> makeNode(
				const SourcePosition& initPoint, Args&&... args)
		{
			static_assert(
					std::is_base_of<Expr, Type>::value ||
							std::is_base_of<Equation, Type>::value,
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
		[[nodiscard]] llvm::Expected<vectorUnique<Equation>> equationList(
				const std::vector<Token>& stopTokens);
		[[nodiscard]] llvm::Expected<std::pair<UniqueEq, UniqueExpr>> ifBrach();
		[[nodiscard]] llvm::Expected<std::pair<std::string, UniqueExpr>> forIndex();

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

		[[nodiscard]] SourcePosition getPosition() const
		{
			return SourcePosition(lexer.getCurrentLine(), lexer.getCurrentColumn());
		}

		private:
		template<typename T>
		bool isError(const ExpectedUnique<T>& t)
		{
			return t;
		}

		bool isError(const llvm::Error& t);
		bool accept(Token t)
		{
			if (current == t)
			{
				next();
				return true;
			}
			return false;
		}
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
		llvm::Expected<bool> expect(Token t);
		void next() { current = lexer.scan(); }
		Lexer<ModelicaStateMachine> lexer;
		Token current;
	};

}	// namespace modelica
