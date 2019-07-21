#pragma once
#include <optional>
#include <string>

#include "modelica/NumbersLexer.hpp"

namespace modelica
{
	enum Token
	{
		None,
		Begin,
		Ident,
		Integer,
		FloatingPoint,
		String,
		Error,

		AlgorithmKeyword,
		AndKeyword,
		AnnotationKeyword,
		BlockKeyword,
		BreakKeyword,
		ClassKeyword,
		ConnectKeyword,
		ConnectorKeyword,
		ConstantKeyword,
		ConstraynedByKeyword,
		DerKeyword,
		DiscreteKeyword,
		EachKeyword,
		ElseKeyword,
		ElseIfKeyword,
		ElseWhenKeyword,
		EncapsulatedKeyword,
		EndKeyword,
		EnumerationKeyword,
		EquationKeyword,
		ExpandableKeyword,
		ExtendsKeyword,
		ExternalKeyword,
		FalseKeyword,
		FinalKeyword,
		FlowKeyword,
		ForKeyword,
		FunctionKeyword,
		IfKeyword,
		ImportKeyword,
		ImpureKeyword,
		InKeyword,
		InitialKeyword,
		InnerKeyword,
		InputKeyword,
		LoopKeyword,
		ModelKeyword,
		NotKeyword,
		OperaptorKeyword,
		OrKeyword,
		OuterKeyword,
		OutputKeyword,
		PackageKeyword,
		ParameterKeyword,
		PartialKeyword,
		ProtectedKeyword,
		PublicKeyword,
		PureKeyword,
		RecordKeyword,
		RedeclareKeyword,
		ReplacableKeyword,
		ReturnKeyword,
		StremKeyword,
		ThenKeyword,
		TrueKeyword,
		TypeKeyword,
		WhenKeyword,
		WhileKeyword,
		WhithinKeyword,

		Multiply,
		Division,
		Dot,
		Plus,
		Minus,
		LessThan,
		Equal,
		GreaterThan,
		Colons,
		LPar,
		RPar,
		LSquare,
		RSquare,
		LCurly,
		RCurly,

		End
	};

	/**
	 * State machine is the state machine of the modelica language.
	 * It implements the interface required by lexer.
	 */
	class ModelicaStateMachine
	{
		public:
		using Token = modelica::Token;

		ModelicaStateMachine(char first)
				: state(Normal),
					current('\0'),
					next(first),
					currentToken(Token::Begin),
					lastIdentifier(""),
					lastString(""),
					lineNumber(1),
					columnNumber(0)
		{
		}

		/**
		 * The possibles state of the machine.
		 */
		enum State
		{
			Normal,
			ParsingId,
			ParsingQId,
			ParsingIdBackSlash,
			ParsingBackSlash,
			ParsingNum,
			ParsingFloat,
			ParsingFloatExponentialSign,
			ParsingFloatExponent,
			ParsingComment,
			EndOfComment,
			ParsingLineComment,
			ParsingString,
			End
		};

		/**
		 * Returns the last seen token. Begin if none was seen.
		 */
		[[nodiscard]] Token getCurrent() const { return currentToken; }
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
		 * Returns the last string seen, or the one being built if the machine is in
		 * the process of recognizing one.
		 */
		[[nodiscard]] const std::string& getLastString() const
		{
			return lastString;
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

		/**
		 * Returns the last bool found
		 */
		[[nodiscard]] bool getLastBool() const { return lastBool; }

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

		template<State s>
		Token scan();

		State state;
		char current;
		char next;
		Token currentToken;
		std::string lastIdentifier;
		FloatLexer<defaultBase> lastNum;
		std::string lastString;
		bool lastBool;

		int lineNumber;
		int columnNumber;

		std::string error;
	};
}	// namespace modelica
