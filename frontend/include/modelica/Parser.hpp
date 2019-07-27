#pragma once
#include <memory>

#include "llvm/Support/Allocator.h"
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
					std::is_base_of<Expr, Type>::value, "Type was not part of AST");
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
		[[nodiscard]] ExpectedUnique<ExprList> expressionList();
		[[nodiscard]] ExpectedUnique<ArrayConstructorExpr> arrayArguments();
		[[nodiscard]] llvm::Expected<std::pair<std::string, UniqueExpr>> forIndex();

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
