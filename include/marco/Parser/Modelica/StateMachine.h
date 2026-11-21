#ifndef MARCO_PARSER_MODELICA_STATEMACHINE_H
#define MARCO_PARSER_MODELICA_STATEMACHINE_H

#include "marco/Lexer/Lexer.h"
#include "marco/Lexer/Location.h"
#include "marco/Parser/Modelica/Token.h"
#include <memory>
#include <optional>
#include <string>

namespace marco::lexer {
template <>
struct TokenTraits<parser::modelica::Token> {
  static parser::modelica::Token getEOFToken() {
    return parser::modelica::Token(parser::modelica::TokenKind::EndOfFile);
  }
};
} // namespace marco::lexer

namespace marco::parser::modelica {
/// State machine recognizing the tokens of the Modelica language, to be
/// driven by 'lexer::Lexer'. Comments are skipped, and malformed input is
/// reported by 'Error' tokens whose text describes the problem.
///
/// Each step consumes one character, whose token is completed by looking at
/// the following character.
class StateMachine {
public:
  using Token = ::marco::parser::modelica::Token;

  enum class State {
    Normal,
    SingleLineComment,
    MultiLineCommentOpening,
    MultiLineComment,
    MultiLineCommentClosing,
    Identifier,
    QuotedIdentifier,
    QuotedIdentifierEscape,
    String,
    StringEscape,
    NumberInteger,
    NumberFraction,
    NumberExponentMarker,
    NumberExponentSign,
    NumberExponent,
    SymbolSecondCharacter
  };

  StateMachine(const std::shared_ptr<SourceFile> &file, char first);

protected:
  /// Feed a character to the state machine. Returns std::nullopt if no
  /// token has been completed, the 'Error' token if the input is malformed,
  /// and the 'EndOfFile' token once '\0' is reached.
  std::optional<Token> step(char c);

private:
  std::optional<Token> scanNormal();
  std::optional<Token> scanSymbol();
  std::optional<Token> continueNumber();

  /// Start a new token at the current character.
  void beginToken();

  Token makeToken(TokenKind kind) const;
  Token makeError(llvm::StringRef message) const;
  Token makeNumber() const;

private:
  State state = State::Normal;

  /// The character being consumed and the following one.
  char current = '\0';
  char next;

  SourcePosition currentPosition;
  SourcePosition nextPosition;

  /// Remaining bytes of the byte order mark.
  llvm::StringRef byteOrderMark;

  SourcePosition tokenBegin;
  std::string text;
  bool isReal = false;

  /// Kind of the symbol whose second character is to be consumed.
  TokenKind symbolKind = TokenKind::Error;
};
} // namespace marco::parser::modelica

#endif // MARCO_PARSER_MODELICA_STATEMACHINE_H
