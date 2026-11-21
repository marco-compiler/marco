#ifndef MARCO_PARSER_MODELICA_TOKEN_H
#define MARCO_PARSER_MODELICA_TOKEN_H

#include "marco/Lexer/Location.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <variant>

namespace marco::parser::modelica {
enum class TokenKind {
  // Control tokens.
  Begin,
  EndOfFile,
  Error,

  // Tokens with a value.
  Identifier,
  Integer,
  Real,
  String,

  // Keywords.
  Algorithm,
  And,
  Annotation,
  Block,
  Break,
  Class,
  Connect,
  Connector,
  Constant,
  ConstrainedBy,
  Der,
  Discrete,
  Each,
  Else,
  ElseIf,
  ElseWhen,
  Encapsulated,
  End,
  Enumeration,
  Equation,
  Expandable,
  Extends,
  External,
  False,
  Final,
  Flow,
  For,
  Function,
  If,
  Import,
  Impure,
  In,
  Initial,
  Inner,
  Input,
  Loop,
  Model,
  Not,
  Operator,
  Or,
  Outer,
  Output,
  Package,
  Parameter,
  Partial,
  Protected,
  Public,
  Pure,
  Record,
  Redeclare,
  Replaceable,
  Return,
  Stream,
  Then,
  Time,
  True,
  Type,
  When,
  While,
  Within,

  // Symbols.
  Plus,
  PlusEW,
  Minus,
  MinusEW,
  Star,
  StarEW,
  Slash,
  SlashEW,
  Caret,
  CaretEW,
  Equal,
  EqualEqual,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  Assignment,
  Comma,
  Semicolon,
  Colon,
  Dot,
  LParen,
  RParen,
  LSquare,
  RSquare,
  LCurly,
  RCurly
};

/// Get the textual representation of a kind of token, as used in
/// diagnostics.
llvm::StringRef toString(TokenKind kind);

class Token {
public:
  explicit Token(TokenKind kind, SourceRange location = SourceRange::unknown());

  Token(TokenKind kind, SourceRange location, llvm::StringRef value);
  Token(TokenKind kind, SourceRange location, int64_t value);
  Token(TokenKind kind, SourceRange location, double value);

  [[nodiscard]] TokenKind getKind() const;

  template <TokenKind Kind>
  [[nodiscard]] bool isa() const {
    return kind == Kind;
  }

  [[nodiscard]] SourceRange getLocation() const;

  /// Get the text of identifiers (with the quotes of quoted identifiers), the
  /// resolved value of string literals, or the message of errors.
  [[nodiscard]] std::string getString() const;

  [[nodiscard]] int64_t getInt() const;

  [[nodiscard]] double getFloat() const;

  /// Get a description of the token for diagnostics.
  [[nodiscard]] std::string describe() const;

private:
  TokenKind kind;
  SourceRange location;
  std::variant<std::string, int64_t, double> value;
};
} // namespace marco::parser::modelica

#endif // MARCO_PARSER_MODELICA_TOKEN_H
