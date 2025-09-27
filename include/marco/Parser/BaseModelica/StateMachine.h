#ifndef MARCO_PARSER_BASEMODELICA_MODELICASTATEMACHINE_H
#define MARCO_PARSER_BASEMODELICA_MODELICASTATEMACHINE_H

#include "marco/Parser/BaseModelica/Token.h"
#include "marco/Parser/FloatLexer.h"
#include "marco/Parser/Lexer.h"
#include "marco/Parser/Location.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <string>

namespace marco::lexer {
template <>
struct TokenTraits<parser::bmodelica::TokenKind> {
  static parser::bmodelica::Token getEOFToken() {
    return parser::bmodelica::Token(parser::bmodelica::TokenKind::EndOfFile,
                                    SourceRange::unknown());
  }
};
} // namespace marco::lexer

namespace marco::parser::bmodelica {
/// State machine is the state machine of the modelica language.
/// It implements the interface required by the lexer.
class StateMachine {
public:
  using Token = ::marco::parser::bmodelica::Token;

  /// The possible states of the machine.
  enum class State {
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

  StateMachine(const std::shared_ptr<SourceFile> &file, char first);

public:
  /// Returns the last seen identifier, or the one being built if the
  /// machine is in the process of recognizing one.
  std::string getIdentifier() const;

  /// Returns the last seen string, or the one being built if the machine
  /// is in the process of recognizing one.
  std::string getString() const;

  /// Returns the last seen integer, or the one being built if the machine
  /// is in the process of recognizing one.
  ///
  /// Notice that as soon as a new number is found this value is
  /// overridden, even if it was a float and not a int.
  int64_t getInt() const;

  /// Returns the last float seen, or the one being built if the machine
  /// is in the process of recognizing one.
  ///
  /// Notice that as soon as a new number is found this value is
  /// overridden, even if it was a int and not a float.
  double getFloat() const;

  llvm::StringRef getError() const;

  SourcePosition getCurrentPosition() const;

  SourceRange getTokenPosition() const;

protected:
  /// Feed a character to the state machine. Returns std::nullopt if
  /// the current token has not consumed all the possible character
  /// Returns the 'Error' token if the input was malformed.
  /// Returns the 'EndOfFile' token if '\0' was found.
  std::optional<Token> step(char c);

private:
  /// Move to the next character.
  void advance(char c);

  template <State s>
  std::optional<Token> scan();

  /// Try to scan the next symbol by taking into account both the current
  /// and the next characters. This avoids the need to define custom states
  /// to recognize simple symbols such as '==' or ':='.
  std::optional<Token> trySymbolScan();

  void setTokenBeginPosition();
  void setTokenEndPosition();

  Token makeToken(TokenKind kind);

  template <typename T>
  Token makeToken(TokenKind kind, T value) {
    return {kind, SourceRange(beginPosition, endPosition), std::move(value)};
  }

private:
  llvm::StringMap<TokenKind> reservedKeywords;
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
} // namespace marco::parser::bmodelica

#endif // MARCO_PARSER_BASEMODELICA_MODELICASTATEMACHINE_H
