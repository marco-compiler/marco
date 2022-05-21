#ifndef MARCO_VARIABLEFILTER_LEXERSTATEMACHINE_H
#define MARCO_VARIABLEFILTER_LEXERSTATEMACHINE_H

#include "marco/VariableFilter/Token.h"
#include "marco/Utils/NumbersLexer.h"
#include "llvm/ADT/StringMap.h"
#include <map>

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

      LexerStateMachine(char first);

      /// Returns the last seen token, or 'Begin' if none was seen.
      Token getCurrent() const;

      size_t getCurrentLine() const;

      size_t getCurrentColumn() const;

      size_t getTokenStartLine() const;

      size_t getTokenStartColumn() const;

      size_t getTokenEndLine() const;

      size_t getTokenEndColumn() const;

      /// Returns the last seen identifier, or the one being built if the machine
      /// is in the process of recognizing one.
      const std::string& getLastIdentifier() const;

      /// Returns the last seen string, or the one being built if the machine is in
      /// the process of recognizing one.
      const std::string& getLastRegex() const;

      /// Returns the last seen integer, or the one being built if the machine is
      /// in the process of recognizing one.
      ///
      /// Notice that as soon as a new number is found this value is overridden,
      /// even if it was a float and not a int.
      long getLastInt() const;

      /// Returns the string associated to the last Error token found.
      const std::string& getLastError() const;

    protected:
      /// Feeds a character to the state machine, returns 'None' if
      /// the current token has not eaten all the possible character
      /// Returns 'Error' if the input was malformed.
      /// Returns 'End' if '\0' was found.
      Token step(char c);

    private:
      /// Updates column and line number, as well as current and next char.
      void advance(char c);

      Token stringToToken(llvm::StringRef str) const;

      Token charToToken(char c) const;

      template<State s>
      Token scan();

      /// Try to scan the next symbol by taking into account both the current and the
      /// next characters. This avoids the need to define custom states to recognize
      /// simple symbols such as '==' or ':='.
      Token tryScanSymbol();

      void setTokenStartPosition();

      void setTokenEndPosition();

    private:
      State state;
      char current;
      char next;
      Token currentToken;
      std::string lastIdentifier;
      IntegerLexer<10> lastNum;
      std::string lastRegex;

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

  template<>
  Token LexerStateMachine::scan<LexerStateMachine::State::ParsingId>();

  template<>
  Token LexerStateMachine::scan<LexerStateMachine::State::ParsingNum>();

  template<>
  Token LexerStateMachine::scan<LexerStateMachine::State::ParsingRegex>();

  template<>
  Token LexerStateMachine::scan<LexerStateMachine::State::Normal>();
}

#endif // MARCO_VARIABLEFILTER_LEXERSTATEMACHINE_H
