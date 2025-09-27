#ifndef MARCO_PARSER_BASEMODELICA_TOKEN_H
#define MARCO_PARSER_BASEMODELICA_TOKEN_H

#include "marco/Parser/Location.h"
#include "llvm/ADT/StringRef.h"
#include <iostream>
#include <string>
#include <variant>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace marco::parser::bmodelica {
enum class TokenKind {
  // Control tokens.
  None,
  Begin,
  EndOfFile,
  Error,

  // Placeholders.
  // The actual values are contained within the lexer.
  Identifier,
  Integer,
  FloatingPoint,
  String,

  // Reserved language keywords.
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
  Product,
  ProductEW,
  Division,
  DivisionEW,
  Pow,
  PowEW,
  Dot,
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  Comma,
  Semicolon,
  Colon,
  LPar,
  RPar,
  LSquare,
  RSquare,
  LCurly,
  RCurly,
  EqualityOperator,
  AssignmentOperator
};

std::string toString(TokenKind obj);

std::ostream &operator<<(std::ostream &os, const TokenKind &obj);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TokenKind &obj);

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

  [[nodiscard]] std::string getString() const;

  [[nodiscard]] int64_t getInt() const;

  [[nodiscard]] double getFloat() const;

private:
  TokenKind kind;
  SourceRange location;
  std::variant<std::string, int64_t, double> value;
};

std::string toString(const Token &obj);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Token &obj);
} // namespace marco::parser::bmodelica

#endif // MARCO_PARSER_BASEMODELICA_TOKEN_H
