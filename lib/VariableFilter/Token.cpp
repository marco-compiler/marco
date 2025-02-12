#include "marco/VariableFilter/Token.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco;
using namespace ::marco::vf;

namespace marco::vf {
std::string toString(TokenKind token) {
  switch (token) {
  case TokenKind::Begin:
    return "Begin";

  case TokenKind::EndOfFile:
    return "EOF";

  case TokenKind::Error:
    return "Error";

  case TokenKind::Integer:
    return "Integer";

  case TokenKind::Identifier:
    return "Identifier";

  case TokenKind::Regex:
    return "Regex";

  case TokenKind::LPar:
    return "(";

  case TokenKind::RPar:
    return ")";

  case TokenKind::LSquare:
    return "[";

  case TokenKind::RSquare:
    return "]";

  case TokenKind::Comma:
    return ",";

  case TokenKind::Semicolons:
    return ";";

  case TokenKind::Colons:
    return ":";

  case TokenKind::Dollar:
    return "$";

  case TokenKind::DerKeyword:
    return "der";
  }

  return "[Unexpected]";
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TokenKind &obj) {
  return os << toString(obj);
}

Token::Token(TokenKind kind, SourceRange location)
    : kind(kind), location(std::move(location)) {}

Token::Token(TokenKind kind, SourceRange location, llvm::StringRef value)
    : kind(kind), location(std::move(location)), value(value.str()) {}

Token::Token(TokenKind kind, SourceRange location, int64_t value)
    : kind(kind), location(std::move(location)), value(value) {}

TokenKind Token::getKind() const { return kind; }

SourceRange Token::getLocation() const { return location; }

std::string Token::getString() const { return std::get<std::string>(value); }

int64_t Token::getInt() const { return std::get<int64_t>(value); }

std::string toString(const Token &obj) { return toString(obj.getKind()); }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Token &obj) {
  return os << obj.getKind();
}
} // namespace marco::vf
