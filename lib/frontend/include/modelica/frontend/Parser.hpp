#pragma once

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <memory>
#include <modelica/frontend/ClassContainer.hpp>
#include <modelica/frontend/Constant.hpp>
#include <modelica/frontend/Equation.hpp>
#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/ForEquation.hpp>
#include <modelica/frontend/LexerStateMachine.hpp>
#include <modelica/frontend/Member.hpp>
#include <modelica/frontend/ParserErrors.hpp>
#include <modelica/frontend/ReferenceAccess.hpp>
#include <modelica/frontend/Type.hpp>
#include <modelica/frontend/TypePrefix.hpp>
#include <modelica/utils/Lexer.hpp>
#include <optional>

namespace modelica
{
	/**
	 * The parser encapsulates the lexer but not the memory where the string we
	 * are reading is held. It exposes parts of the grammatical rules that are
	 * available in the grammar (can be found at page ~ 265 of the 3.4 doc).
	 */
	class Parser
	{
		public:
		Parser(std::string filename, const std::string& source);
		Parser(const std::string& source);
		Parser(const char* source);

		/**
		 * Return the current position in the source stream.
		 */
		[[nodiscard]] SourcePosition getPosition() const;

		[[nodiscard]] Token getCurrentToken() const;

		llvm::Expected<ClassContainer> classDefinition();
		llvm::Expected<Expression> primary();
		llvm::Expected<Expression> factor();
		llvm::Expected<std::optional<Expression>> modification();
		llvm::Expected<Expression> term();
		llvm::Expected<Type> typeSpecifier();
		llvm::Expected<llvm::SmallVector<ForEquation, 3>> forEquationBody(
				int nestingLevel);
		llvm::Expected<Expression> arithmeticExpression();

		llvm::Expected<llvm::SmallVector<size_t, 3>> arrayDimensions();
		llvm::Expected<llvm::SmallVector<Member, 3>> elementList();

		llvm::Expected<TypePrefix> typePrefix();
		llvm::Expected<Member> element(bool publicSection = true);
		std::optional<OperationKind> relationalOperator();

		llvm::Expected<Expression> logicalTerm();
		llvm::Expected<Expression> logicalExpression();

		llvm::Expected<Equation> equation();
		llvm::Expected<llvm::SmallVector<ForEquation, 3>> forEquation(
				int nestingLevel);

		llvm::Expected<std::pair<llvm::SmallVector<Equation, 3>, llvm::SmallVector<ForEquation, 3>>> equationSection();
		llvm::Expected<Expression> expression();
		llvm::Expected<Expression> logicalFactor();
		llvm::Expected<Expression> relation();
		llvm::Expected<Expression> componentReference();
		llvm::Expected<llvm::SmallVector<Expression, 3>> functionCallArguments();
		llvm::Expected<std::vector<Expression>> arraySubscript();

		llvm::Expected<Algorithm> algorithmSection();
		llvm::Expected<Statement> statement();
		llvm::Expected<AssignmentStatement> assignmentStatement();
		llvm::Expected<IfStatement> ifStatement();
		llvm::Expected<ForStatement> forStatement();
		llvm::Expected<WhileStatement> whileStatement();
		llvm::Expected<WhenStatement> whenStatement();

		llvm::Expected<Tuple> outputExpressionList();

		private:
		/**
		 * Read the next token.
		 */
		void next();

		/**
		 * Regular accept: if the current token is t then the next one will be read
		 * and true will be returned, else false.
		 */
		bool accept(Token t);
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

		llvm::Expected<bool> expect(Token t);

		/**
		 * Unfortunately the grammar in the specification is written in such a
		 * way that there is a single point where you need a two lookahead
		 * to tell the difference between a component reference and a
		 * named argument. Instead of adding a real two lookahead or to completely
		 * change factor the grammar we can provide the ability to undo the last
		 * accept. If you need to implement that particular case, use this.
		 */
		void undoScan(Token t);
		const std::string filename;
		Lexer<ModelicaStateMachine> lexer;
		Token current;
		Token undo;
	};
}	 // namespace modelica
