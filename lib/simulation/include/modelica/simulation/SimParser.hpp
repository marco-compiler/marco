#pragma once
#include <memory>

#include "modelica/simulation/SimErrors.hpp"
#include "modelica/simulation/SimLexerStateMachine.hpp"
#include "modelica/utils/Lexer.hpp"
#include "modelica/utils/SourceRange.hpp"

namespace modelica
{
	class SimParser
	{
		public:
		SimParser(const std::string& source)
				: lexer(source), current(lexer.scan()), undo(SimToken::End)
		{
		}

		SimParser(const char* source)
				: lexer(source), current(lexer.scan()), undo(SimToken::End)
		{
		}

		/**
		 * Return the current position in the source stream
		 */
		[[nodiscard]] SourcePosition getPosition() const
		{
			return SourcePosition(lexer.getCurrentLine(), lexer.getCurrentColumn());
		}

		[[nodiscard]] llvm::Expected<SimExp> expression();
		[[nodiscard]] llvm::Expected<SimConst<bool>> boolVector();
		[[nodiscard]] llvm::Expected<SimConst<int>> intVector();
		[[nodiscard]] llvm::Expected<SimConst<float>> floatVector();
		[[nodiscard]] llvm::Expected<std::string> reference();
		[[nodiscard]] llvm::Expected<SimCall> call();
		[[nodiscard]] llvm::Expected<SimType> type();
		[[nodiscard]] llvm::Expected<std::vector<size_t>> typeDimensions();
		[[nodiscard]] llvm::Expected<std::vector<SimExp>> args();
		[[nodiscard]] llvm::Expected<std::tuple<SimExpKind, std::vector<SimExp>>>
		operation();
		[[nodiscard]] llvm::Expected<std::tuple<std::string, SimExp>> statement();
		[[nodiscard]] llvm::Expected<llvm::StringMap<SimExp>> initSection();
		[[nodiscard]] llvm::Expected<float> floatingPoint();
		[[nodiscard]] llvm::Expected<int> integer();
		[[nodiscard]] llvm::Expected<llvm::StringMap<SimExp>> updateSection();
		[[nodiscard]] llvm::Expected<
				std::tuple<llvm::StringMap<SimExp>, llvm::StringMap<SimExp>>>
		simulation();

		[[nodiscard]] SimToken getCurrentSimToken() const { return current; }

		private:
		/**
		 * regular accept, if the current token it then the next one will be read
		 * and true will be returned, else false.
		 */
		bool accept(SimToken t)
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
		template<SimToken t>
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
		llvm::Expected<bool> expect(SimToken t);

		/**
		 * reads the next token
		 */
		void next() { current = lexer.scan(); }

		Lexer<SimLexerStateMachine> lexer;
		SimToken current;
		SimToken undo;
	};

}	 // namespace modelica
