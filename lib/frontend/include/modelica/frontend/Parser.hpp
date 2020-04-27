#pragma once
#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/LexerStateMachine.hpp"
#include "modelica/frontend/ParserErrors.hpp"
#include "modelica/frontend/ReferenceAccess.hpp"
#include "modelica/utils/Lexer.hpp"

namespace modelica
{
	/**
	 * The parser encapsulates the lexer but not he memory where string we are
	 * reading is held. It expones parts of the grammatical rules that are
	 * avialable in the grammar (can be found at page ~ 265 of the 3.4 doc).
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
		 * Return the current position in the source stream
		 */
		[[nodiscard]] SourcePosition getPosition() const
		{
			return SourcePosition(lexer.getCurrentLine(), lexer.getCurrentColumn());
		}

		[[nodiscard]] Token getCurrentToken() const { return current; }

		llvm::Expected<Expression> primary();
		llvm::Expected<Expression> factor();
		llvm::Expected<Expression> term();
		llvm::Expected<Expression> arithmeticExpression();
		std::optional<OperationKind> relationalOperator();

		llvm::Expected<Expression> logicalTerm();
		llvm::Expected<Expression> logicalExpression();

		llvm::Expected<Expression> expression();
		llvm::Expected<Expression> logicalFactor();
		llvm::Expected<Expression> relation();
		llvm::Expected<Expression> componentReference();
		llvm::Expected<llvm::SmallVector<Expression, 3>> functionCallArguments();
		llvm::Expected<std::vector<Expression>> arraySubscript();

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

		llvm::Expected<bool> expect(Token t);

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

}	 // namespace modelica
