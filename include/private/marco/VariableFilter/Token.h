#ifndef MARCO_VARIABLEFILTER_TOKEN_H
#define MARCO_VARIABLEFILTER_TOKEN_H

#include "marco/Parser/Location.h"
#include <string>
#include <variant>

namespace llvm {
class raw_ostream;
}

namespace marco::vf {
enum class TokenKind {
  // Control tokens
  Begin,
  EndOfFile,
  Error,

  // Placeholders
  Integer,
  Identifier,
  Regex,

  // Symbols
  LPar,
  RPar,
  LSquare,
  RSquare,
  Dollar,
  Comma,
  Semicolons,
  Colons,

  // Keywords
  DerKeyword
};

std::string toString(TokenKind obj);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TokenKind &obj);

class Token {
public:
  explicit Token(TokenKind kind, SourceRange location = SourceRange::unknown());

  Token(TokenKind kind, SourceRange location, llvm::StringRef value);
  Token(TokenKind kind, SourceRange location, int64_t value);

  [[nodiscard]] TokenKind getKind() const;

  template <TokenKind Kind>
  [[nodiscard]] bool isa() const {
    return kind == Kind;
  }

  [[nodiscard]] SourceRange getLocation() const;

  [[nodiscard]] std::string getString() const;

  [[nodiscard]] int64_t getInt() const;

private:
  TokenKind kind;
  SourceRange location;
  std::variant<std::string, int64_t> value;
};

std::string toString(const Token &obj);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Token &obj);
} // namespace marco::vf

#endif // MARCO_VARIABLEFILTER_TOKEN_H
