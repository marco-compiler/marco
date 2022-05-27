#ifndef MARCO_PARSER_LEXER_H
#define MARCO_PARSER_LEXER_H

#include "marco/Diagnostic/Location.h"
#include "marco/Parser/Lexer.h"
#include "marco/Parser/FloatLexer.h"
#include "marco/Parser/Token.h"
#include "llvm/ADT/StringMap.h"
#include <string>

namespace marco::parser
{
	/// State machine is the state machine of the modelica language.
	/// It implements the interface required by the lexer.
  class ModelicaStateMachine
  {
    public:
      using Token = ::marco::parser::Token;

      /// The possible states of the machine.
      enum class State
      {
        Normal,
        End,

        ParsingIdentifier,
        ParsingQIdentifier,
        ParsingNumber,
        ParsingFloat,
        ParsingFloatExponentSign,
        ParsingFloatExponent,
        ParsingString,
        ParsingSingleLineComment,
        ParsingMultiLineComment,
        ParsingBackSlash,
        ParsingQIdentifierBackSlash,
        ParsingElementWiseSymbol,
        ParsingEqualSymbol,
        ParsingLessSymbol,
        ParsingGreaterSymbol,
        ParsingColonSymbol,

        IgnoredCharacter
      };

      ModelicaStateMachine(llvm::StringRef file, char first);

		  /// Returns the last seen identifier, or the one being built if the machine
		  /// is in the process of recognizing one.
      std::string getIdentifier() const;

		  /// Returns the last seen string, or the one being built if the machine is in
		  /// the process of recognizing one.
      std::string getString() const;

		  /// Returns the last seen integer, or the one being built if the machine is
		  /// in the process of recognizing one.
		  ///
		  /// Notice that as soon as a new number is found this value is overridden,
		  /// even if it was a float and not a int.
      int64_t getInt() const;

		  /// Returns the last float seen, or the one being built if the machine is in
		  /// the process of recognizing one.
		  ///
		  /// Notice that as soon as a new number is found this value is overridden,
		  /// even if it was a int and not a float.
      double getFloat() const;

      std::string getError() const;

      SourcePosition getCurrentPosition() const;

      SourceRange getTokenPosition() const;

    protected:
      Token getNoneToken() const;

      /// Feed a character to the state machine. Returns 'None' if
      /// the current token has not consumed all the possible character
      /// Returns 'Error' if the input was malformed.
      /// Returns 'EndOfFile' if '\0' was found.
      Token step(char c);

    private:
		  /// Move to the next character.
      void advance(char c);

      template<State s>
      Token scan();

      /// Try to scan the next symbol by taking into account both the current and the
      /// next characters. This avoids the need to define custom states to recognize
      /// simple symbols such as '==' or ':='.
      Token trySymbolScan();

      void setTokenBeginPosition();
      void setTokenEndPosition();

    private:
      llvm::StringMap<Token> reservedKeywords;
      State state;

      char current;
      char next;

      std::string identifier;
      std::string stringValue;
      FloatLexer<10> numberLexer;
      std::string error;

      SourcePosition currentPosition;
      SourcePosition beginPosition;
      SourcePosition endPosition;
  };
}

#endif // MARCO_PARSER_LEXER_H
