#include "marco/Parser/Token.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco;
using namespace ::marco::parser;

namespace marco::parser
{
  std::string toString(TokenKind obj)
  {
    switch (obj) {
      case TokenKind::None:
        return "none";

      case TokenKind::Begin:
        return "begin";

      case TokenKind::EndOfFile:
        return "EOF";

      case TokenKind::Error:
        return "error";

      case TokenKind::Identifier:
        return "identifier";

      case TokenKind::Integer:
        return "integer";

      case TokenKind::FloatingPoint:
        return "floating point";

      case TokenKind::String:
        return "string";

      case TokenKind::Algorithm:
        return "algorithm";

      case TokenKind::And:
        return "and";

      case TokenKind::Annotation:
        return "annotation";

      case TokenKind::Block:
        return "block";

      case TokenKind::Break:
        return "break";

      case TokenKind::Class:
        return "class";

      case TokenKind::Connect:
        return "connect";

      case TokenKind::Connector:
        return "connector";

      case TokenKind::Constant:
        return "constant";

      case TokenKind::ConstrainedBy:
        return "constrainedby";

      case TokenKind::Der:
        return "der";

      case TokenKind::Discrete:
        return "discrete";

      case TokenKind::Each:
        return "each";

      case TokenKind::Else:
        return "else";

      case TokenKind::ElseIf:
        return "elseif";

      case TokenKind::ElseWhen:
        return "elsewhen";

      case TokenKind::Encapsulated:
        return "encapsulated";

      case TokenKind::End:
        return "end";

      case TokenKind::Enumeration:
        return "enumeration";

      case TokenKind::Equation:
        return "equation";

      case TokenKind::Expandable:
        return "expandable";

      case TokenKind::Extends:
        return "extends";

      case TokenKind::External:
        return "external";

      case TokenKind::False:
        return "false";

      case TokenKind::Final:
        return "final";

      case TokenKind::Flow:
        return "flow";

      case TokenKind::For:
        return "for";

      case TokenKind::Function:
        return "function";

      case TokenKind::If:
        return "if";

      case TokenKind::Import:
        return "import";

      case TokenKind::Impure:
        return "impure";

      case TokenKind::In:
        return "in";

      case TokenKind::Initial:
        return "initial";

      case TokenKind::Inner:
        return "inner";

      case TokenKind::Input:
        return "input";

      case TokenKind::Loop:
        return "loop";

      case TokenKind::Model:
        return "model";

      case TokenKind::Not:
        return "not";

      case TokenKind::Operator:
        return "operator";

      case TokenKind::Or:
        return "or";

      case TokenKind::Outer:
        return "outer";

      case TokenKind::Output:
        return "output";

      case TokenKind::Package:
        return "package";

      case TokenKind::Parameter:
        return "parameter";

      case TokenKind::Partial:
        return "partial";

      case TokenKind::Protected:
        return "protected";

      case TokenKind::Public:
        return "public";

      case TokenKind::Pure:
        return "pure";

      case TokenKind::Record:
        return "record";

      case TokenKind::Redeclare:
        return "redeclare";

      case TokenKind::Replaceable:
        return "replaceable";

      case TokenKind::Return:
        return "return";

      case TokenKind::Stream:
        return "stream";

      case TokenKind::Then:
        return "then";

      case TokenKind::True:
        return "true";

      case TokenKind::Type:
        return "type";

      case TokenKind::When:
        return "when";

      case TokenKind::While:
        return "while";

      case TokenKind::Within:
        return "within";

      case TokenKind::Plus:
        return "+";

      case TokenKind::PlusEW:
        return ".+";

      case TokenKind::Minus:
        return "-";

      case TokenKind::MinusEW:
        return ".-";

      case TokenKind::Product:
        return "*";

      case TokenKind::ProductEW:
        return ".*";

      case TokenKind::Division:
        return "/";

      case TokenKind::DivisionEW:
        return "./";

      case TokenKind::Pow:
        return "^";

      case TokenKind::PowEW:
        return ".^";

      case TokenKind::Dot:
        return ".";

      case TokenKind::Equal:
        return "==";

      case TokenKind::NotEqual:
        return "<>";

      case TokenKind::Less:
        return "<";

      case TokenKind::LessEqual:
        return "<=";

      case TokenKind::Greater:
        return ">";

      case TokenKind::GreaterEqual:
        return ">=";

      case TokenKind::Comma:
        return ",";

      case TokenKind::Semicolon:
        return ";";

      case TokenKind::Colon:
        return ":";

      case TokenKind::LPar:
        return "(";

      case TokenKind::RPar:
        return ")";

      case TokenKind::LSquare:
        return "[";

      case TokenKind::RSquare:
        return "]";

      case TokenKind::LCurly:
        return "{";

      case TokenKind::RCurly:
        return "}";

      case TokenKind::EqualityOperator:
        return "=";

      case TokenKind::AssignmentOperator:
        return ":=";
    }

    llvm_unreachable("Unknown token");
    return "unknown";
  }

  llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TokenKind& obj)
  {
    return os << toString(obj);
  }

  Token::Token(TokenKind kind, SourceRange location)
      : kind(kind), location(std::move(location))
  {
  }

  Token::Token(TokenKind kind, SourceRange location, llvm::StringRef value)
      : kind(kind),
        location(std::move(location)),
        value(value.str())
  {

  }

  Token::Token(TokenKind kind, SourceRange location, int64_t value)
      : kind(kind),
        location(std::move(location)),
        value(value)
  {

  }

  Token::Token(TokenKind kind, SourceRange location, double value)
      : kind(kind),
        location(std::move(location)),
        value(value)
  {
  }

  TokenKind Token::getKind() const
  {
    return kind;
  }

  SourceRange Token::getLocation() const
  {
    return location;
  }

  std::string Token::getString() const
  {
    return std::get<std::string>(value);
  }

  int64_t Token::getInt() const
  {
    return std::get<int64_t>(value);
  }

  double Token::getFloat() const
  {
    return std::get<double>(value);
  }

  std::string toString(const Token& obj)
  {
    return toString(obj.getKind());
  }

  llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Token& obj)
  {
    return os << obj.getKind();
  }
}
