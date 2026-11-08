#ifndef MARCO_PARSER_MODELICA_TOKEN_H
#define MARCO_PARSER_MODELICA_TOKEN_H

#include "marco/Lexer/Location.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace marco::parser::modelica {
enum class TokenKind {
  // Control tokens.
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

struct Token {
  TokenKind kind = TokenKind::EndOfFile;
  SourceRange location = SourceRange::unknown();

  /// Identifiers (with the quotes of quoted identifiers), resolved string
  /// literals, or error messages.
  std::string text;

  int64_t integerValue = 0;
  double realValue = 0;

  bool is(TokenKind other) const { return kind == other; }

  /// Get a description of the token for diagnostics.
  std::string describe() const;
};
} // namespace marco::parser::modelica

#endif // MARCO_PARSER_MODELICA_TOKEN_H
