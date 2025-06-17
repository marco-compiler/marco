#ifndef MARCO_VARIABLEFILTER_LEXERSTATEMACHINE_H
#define MARCO_VARIABLEFILTER_LEXERSTATEMACHINE_H

#include "Token.h"
#include "marco/Parser/IntegerLexer.h"
#include "marco/Parser/Lexer.h"
#include "marco/Parser/Location.h"
#include "llvm/ADT/StringMap.h"
#include <map>
#include <memory>

namespace marco::lexer {
template <>
struct TokenTraits<vf::TokenKind> {
  static vf::TokenKind getEOFToken() { return vf::TokenKind::EndOfFile; }
};
} // namespace marco::lexer

namespace marco::vf {
/// State machine is the state machine of the variable filter grammar.
/// It implements the interface required by lexer.
class LexerStateMachine {
public:
  using Token = ::marco::vf::Token;

  /// The possible states of the machine.
  enum class State {
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
  std::string getIdentifier() const;

  /// Returns the last seen string, or the one being built if the machine
  /// is in the process of recognizing one.
  std::string getRegex() const;

  /// Returns the last seen integer, or the one being built if the machine
  /// is in the process of recognizing one.
  ///
  /// Notice that as soon as a new number is found this value is
  /// overridden, even if it was a float and not a int.
  int64_t getInt() const;

  llvm::StringRef getError() const;

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

  TokenKind stringToToken(llvm::StringRef str) const;

  TokenKind charToToken(char c) const;

  template <State s>
  std::optional<Token> scan();

  Token trySymbolScan();

  void setTokenBeginPosition();
  void setTokenEndPosition();

  Token makeToken(TokenKind kind);

  template <typename T>
  Token makeToken(TokenKind kind, T value) {
    return {kind, SourceRange(beginPosition, endPosition), std::move(value)};
  }

private:
  State state;

  char current;
  char next;

  std::string identifier;
  IntegerLexer<10> numberLexer;
  std::string regex;

  SourcePosition currentPosition;
  SourcePosition beginPosition;
  SourcePosition endPosition;

  std::string error;
  llvm::StringMap<TokenKind> keywordMap;
  std::map<char, TokenKind> symbols;
};

template <>
std::optional<Token>
LexerStateMachine::scan<LexerStateMachine::State::ParsingId>();

template <>
std::optional<Token>
LexerStateMachine::scan<LexerStateMachine::State::ParsingNum>();

template <>
std::optional<Token>
LexerStateMachine::scan<LexerStateMachine::State::ParsingRegex>();

template <>
std::optional<Token>
LexerStateMachine::scan<LexerStateMachine::State::Normal>();
} // namespace marco::vf

#endif // MARCO_VARIABLEFILTER_LEXERSTATEMACHINE_H
