#pragma once

#include <map>
#include <string>

#include "llvm/ADT/StringMap.h"
#include "modelica/utils/NumbersLexer.hpp"

namespace modelica
{
	enum class ModToken
	{
		None,
		Begin,
		Ident,
		Float,
		Bool,
		Integer,
		Error,
		CallKeyword,
		InitKeyword,
		UpdateKeyword,
		BoolKeyword,
		IntKeyword,
		FloatKeyword,

		LPar,
		Comma,
		RPar,
		LSquare,
		RSquare,
		LCurly,
		RCurly,

		Multiply,
		Division,
		Plus,
		Minus,
		OperatorEqual,
		LessThan,
		LessEqual,
		GreaterThan,
		GreaterEqual,
		Modulo,
		Exponential,
		Assign,
		Ternary,
		Not,

		End
	};

	std::string tokenToString(ModToken token);

	class ModLexerStateMachine
	{
		public:
		using Token = modelica::ModToken;
		ModLexerStateMachine(char first);

		enum class State
		{
			Normal,
			ParsingId,
			ParsingNum,
			ParsingFloat,
			ParsingFloatExponentialSign,
			ParsingFloatExponent,
			ParsingComment,
			EndOfComment,
			ParsingLineComment,
			IgnoreNextChar,
			End
		};

		/**
		 * Returns the last seen token. Begin if none was seen.
		 */
		[[nodiscard]] ModToken getCurrent() const { return currentToken; }
		[[nodiscard]] int getCurrentLine() const { return lineNumber; }
		[[nodiscard]] int getCurrentColumn() const { return columnNumber; }
		/**
		 * Returns the last identifier seen, or the one being built if the machine
		 * is in the process of recognizing one.
		 */
		[[nodiscard]] const std::string& getLastIdentifier() const
		{
			return lastIdentifier;
		}

		/**
		 * Returns the last int seen, or the one being built if the machine is in
		 * the process of recognizing one.
		 *
		 * Notice that as soon as a new number is found this value is overridden,
		 * even if it was a float and not a int
		 */
		[[nodiscard]] int getLastInt() const { return lastNum.getUpperPart(); }
		/**
		 * Returns the last float seen, or the one being built if the machine is in
		 * the process of recognizing one.
		 *
		 * Notice that as soon as a new number is found this value is overridden,
		 * even if it was a int and not a float
		 *
		 */
		[[nodiscard]] double getLastFloat() const { return lastNum.get(); }

		/**
		 * Returns the string associated to the last Error token found
		 */
		[[nodiscard]] const std::string& getLastError() const { return error; }

		protected:
		/**
		 * Feeds a character to the state machine, returns None if
		 * the current token has not eaten all the possible character
		 * Returns Error if the input was illformed.
		 * Returns End if \0 was found.
		 */
		Token step(char c);

		private:
		/**
		 * Updates column and line number, as well as current and next char.
		 */
		void advance(char c)
		{
			current = next;
			next = c;
			columnNumber++;

			if (current == '\n')
			{
				columnNumber = 0;
				lineNumber++;
			}
		}
		[[nodiscard]] Token stringToToken(const std::string& lookUp) const;
		[[nodiscard]] Token charToToken(char c) const;

		template<State s>
		Token scan();
		Token tryScanSymbol();

		State state;
		char current;
		char next;
		Token currentToken;
		std::string lastIdentifier;
		FloatLexer<defaultBase> lastNum;

		int lineNumber;
		int columnNumber;

		std::string error;
		llvm::StringMap<Token> keywordMap;
		std::map<char, Token> symbols;
	};
}	 // namespace modelica
