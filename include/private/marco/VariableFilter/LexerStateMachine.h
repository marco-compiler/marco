#ifndef MARCO_VARIABLEFILTER_LEXERSTATEMACHINE_H
#define MARCO_VARIABLEFILTER_LEXERSTATEMACHINE_H

#include "marco/Diagnostic/Location.h"
#include "marco/Parser/Lexer.h"
#include "marco/Parser/IntegerLexer.h"
#include "marco/VariableFilter/Token.h"
#include "llvm/ADT/StringMap.h"
#include <map>
#include <memory>

namespace marco::lexer
{
  template<>
  struct TokenTraits<vf::Token>
  {
    static vf::Token getEOFToken()
    {
      return vf::Token::EndOfFile;
    }
  };
}

namespace marco::vf
{
  /// State machine is the state machine of the variable filter grammar.
  /// It implements the interface required by lexer.
  class LexerStateMachine
  {
    public:
      using Token = ::marco::vf::Token;

      /// The possible states of the machine.
      enum class State
      {
        Normal,
        ParsingId,
        ParsingNum,
        ParsingRegex,
        IgnoreNextChar,
        End
      };

      LexerStateMachine(std::shared_ptr<SourceFile> file, char first);

      /// Returns the last seen identifier, or the one being built if the
      /// machine is in the process of recognizing one.
      const std::string& getLastIdentifier() const;

      /// Returns the last seen string, or the one being built if the machine
      /// is in the process of recognizing one.
      const std::string& getLastRegex() const;

      /// Returns the last seen integer, or the one being built if the machine
      /// is in the process of recognizing one.
      ///
      /// Notice that as soon as a new number is found this value is
      /// overridden, even if it was a float and not a int.
      long getLastInt() const;

      /// Returns the string associated to the last Error token found.
      const std::string& getLastError() const;

      SourcePosition getCurrentPosition() const;

      SourceRange getTokenPosition() const;

    protected:
      /// Feeds a character to the state machine, returns 'None' if the current
      /// token has not eaten all the possible character
      /// Returns the 'Error' token if the input was malformed.
      /// Returns the 'End' token if '\0' was found.
      std::optional<Token> step(char c);

    private:
      /// Updates column and line number, as well as current and next char.
      void advance(char c);

      Token stringToToken(llvm::StringRef str) const;

      Token charToToken(char c) const;

      template<State s>
      std::optional<Token> scan();

      /// Try to scan the next symbol by taking into account both the current
      /// and the next characters. This avoids the need to define custom states
      /// to recognize simple symbols such as '==' or ':='.
      Token tryScanSymbol();

      void setTokenBeginPosition();
      void setTokenEndPosition();

    private:
      State state;

      char current;
      char next;

      std::string lastIdentifier;
      IntegerLexer<10> lastNum;
      std::string lastRegex;

      SourcePosition currentPosition;
      SourcePosition beginPosition;
      SourcePosition endPosition;

      std::string error;
      llvm::StringMap<Token> keywordMap;
      std::map<char, Token> symbols;
  };

  template<>
  std::optional<Token> LexerStateMachine::scan<
      LexerStateMachine::State::ParsingId>();

  template<>
  std::optional<Token> LexerStateMachine::scan<
      LexerStateMachine::State::ParsingNum>();

  template<>
  std::optional<Token> LexerStateMachine::scan<
      LexerStateMachine::State::ParsingRegex>();

  template<>
  std::optional<Token> LexerStateMachine::scan<
      LexerStateMachine::State::Normal>();
}

#endif // MARCO_VARIABLEFILTER_LEXERSTATEMACHINE_H
