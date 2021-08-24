#pragma once

#include <llvm/ADT/StringMap.h>
#include <llvm/Support/raw_ostream.h>
#include <map>
#include <marco/utils/NumbersLexer.hpp>
#include <marco/utils/SourcePosition.h>
#include <optional>
#include <string>

namespace marco::frontend
{
	enum class Token
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
		OperatorKeyword,
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
		ReplaceableKeyword,
		ReturnKeyword,
		StreamKeyword,
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
		ElementWiseMinus,
		ElementWiseSum,
		ElementWiseMultilpy,
		ElementWiseDivision,
		ElementWiseExponential,
		OperatorEqual,
		LessThan,
		LessEqual,
		Equal,
		GreaterThan,
		GreaterEqual,
		Different,
		Colons,
		Semicolons,
		Comma,
		LPar,
		RPar,
		LSquare,
		RSquare,
		LCurly,
		RCurly,
		Exponential,
		Assignment,

		End	 // Notice that keywordEnd is not the end of file token, end token is
				 // end of file token
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Token& obj);

	std::string toString(Token obj);

	/**
	 * State machine is the state machine of the modelica language.
	 * It implements the interface required by lexer.
	 */
	class ModelicaStateMachine
	{
		public:
		using Token = marco::frontend::Token;

		ModelicaStateMachine(char first);

		/**
		 * The possible states of the machine.
		 */
		enum class State
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
			IgnoreNextChar,
			End
		};

		/**
		 * Returns the last seen token, or 'Begin' if none was seen.
		 */
		[[nodiscard]] Token getCurrent() const { return currentToken; }

		[[nodiscard]] size_t getCurrentLine() const { return currentLine; }
		[[nodiscard]] size_t getCurrentColumn() const { return currentColumn; }

		[[nodiscard]] size_t getTokenStartLine() const { return startLine; }
		[[nodiscard]] size_t getTokenStartColumn() const { return startColumn; }

		[[nodiscard]] size_t getTokenEndLine() const { return endLine; }
		[[nodiscard]] size_t getTokenEndColumn() const { return endColumn; }

		/**
		 * Returns the last seen identifier, or the one being built if the machine
		 * is in the process of recognizing one.
		 */
		[[nodiscard]] const std::string& getLastIdentifier() const
		{
			return lastIdentifier;
		}

		/**
		 * Returns the last seen string, or the one being built if the machine is in
		 * the process of recognizing one.
		 */
		[[nodiscard]] const std::string& getLastString() const
		{
			return lastString;
		}

		/**
		 * Returns the last seen integer, or the one being built if the machine is
		 * in the process of recognizing one.
		 *
		 * Notice that as soon as a new number is found this value is overridden,
		 * even if it was a float and not a int.
		 */
		[[nodiscard]] long getLastInt() const { return lastNum.getUpperPart(); }

		/**
		 * Returns the last float seen, or the one being built if the machine is in
		 * the process of recognizing one.
		 *
		 * Notice that as soon as a new number is found this value is overridden,
		 * even if it was a int and not a float.
		 */
		[[nodiscard]] double getLastFloat() const { return lastNum.get(); }

		/**
		 * Returns the string associated to the last Error token found.
		 */
		[[nodiscard]] const std::string& getLastError() const { return error; }

		protected:
		/**
		 * Feeds a character to the state machine, returns 'None' if
		 * the current token has not eaten all the possible character
		 * Returns 'Error' if the input was malformed.
		 * Returns 'End' if '\0' was found.
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
			currentColumn++;

			if (current == '\n')
			{
				currentColumn = 0;
				currentLine++;
			}
		}

		[[nodiscard]] Token stringToToken(const std::string& lookUp) const;
		[[nodiscard]] Token charToToken(char c) const;

		template<State s>
		Token scan();

		Token tryScanSymbol();

		void setTokenStartPosition();
		void setTokenEndPosition();

		State state;
		char current;
		char next;
		Token currentToken;
		std::string lastIdentifier;
		FloatLexer<defaultBase> lastNum;
		std::string lastString;

		size_t currentLine;
		size_t currentColumn;

		size_t startLine;
		size_t startColumn;

		size_t endLine;
		size_t endColumn;

		std::string error;
		llvm::StringMap<Token> keywordMap;
		std::map<char, Token> symbols;
	};
}
