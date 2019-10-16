#pragma once
#include <memory>

#include "modelica/model/ModErrors.hpp"
#include "modelica/model/ModLexerStateMachine.hpp"
#include "modelica/utils/Lexer.hpp"
#include "modelica/utils/SourceRange.hpp"

namespace modelica
{
	class ModParser
	{
		public:
		ModParser(const std::string& source)
				: lexer(source), current(lexer.scan()), undo(ModToken::End)
		{
		}

		ModParser(const char* source)
				: lexer(source), current(lexer.scan()), undo(ModToken::End)
		{
		}

		/**
		 * Return the current position in the source stream
		 */
		[[nodiscard]] SourcePosition getPosition() const
		{
			return SourcePosition(lexer.getCurrentLine(), lexer.getCurrentColumn());
		}

		[[nodiscard]] llvm::Expected<ModExp> expression();
		[[nodiscard]] llvm::Expected<ModConst<bool>> boolVector();
		[[nodiscard]] llvm::Expected<ModConst<int>> intVector();
		[[nodiscard]] llvm::Expected<ModConst<float>> floatVector();
		[[nodiscard]] llvm::Expected<std::string> reference();
		[[nodiscard]] llvm::Expected<ModCall> call();
		[[nodiscard]] llvm::Expected<ModType> type();
		[[nodiscard]] llvm::Expected<std::vector<size_t>> typeDimensions();
		[[nodiscard]] llvm::Expected<std::vector<ModExp>> args();
		[[nodiscard]] llvm::Expected<std::tuple<ModExpKind, std::vector<ModExp>>>
		operation();
		[[nodiscard]] llvm::Expected<std::tuple<std::string, ModExp>> statement();
		[[nodiscard]] llvm::Expected<llvm::StringMap<ModExp>> initSection();
		[[nodiscard]] llvm::Expected<float> floatingPoint();
		[[nodiscard]] llvm::Expected<int> integer();
		[[nodiscard]] llvm::Expected<llvm::StringMap<ModExp>> updateSection();
		[[nodiscard]] llvm::Expected<
				std::tuple<llvm::StringMap<ModExp>, llvm::StringMap<ModExp>>>
		simulation();

		[[nodiscard]] ModToken getCurrentModToken() const { return current; }

		private:
		/**
		 * regular accept, if the current token it then the next one will be read
		 * and true will be returned, else false.
		 */
		bool accept(ModToken t)
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
		template<ModToken t>
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
		llvm::Expected<bool> expect(ModToken t);

		/**
		 * reads the next token
		 */
		void next() { current = lexer.scan(); }

		Lexer<ModLexerStateMachine> lexer;
		ModToken current;
		ModToken undo;
	};

}	 // namespace modelica
