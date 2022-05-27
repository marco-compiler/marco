#include "marco/Parser/Token.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco;
using namespace ::marco::parser;

namespace marco::parser
{
  std::string toString(Token obj)
  {
    switch (obj) {
      case Token::None:
        return "none";

      case Token::Begin:
        return "begin";

      case Token::EndOfFile:
        return "EOF";

      case Token::Error:
        return "error";

      case Token::Identifier:
        return "identifier";

      case Token::Integer:
        return "integer";

      case Token::FloatingPoint:
        return "floating point";

      case Token::String:
        return "string";

      case Token::Algorithm:
        return "algorithm";

      case Token::And:
        return "and";

      case Token::Annotation:
        return "annotation";

      case Token::Block:
        return "block";

      case Token::Break:
        return "break";

      case Token::Class:
        return "class";

      case Token::Connect:
        return "connect";

      case Token::Connector:
        return "connector";

      case Token::Constant:
        return "constant";

      case Token::ConstrainedBy:
        return "constrainedby";

      case Token::Der:
        return "der";

      case Token::Discrete:
        return "discrete";

      case Token::Each:
        return "each";

      case Token::Else:
        return "else";

      case Token::ElseIf:
        return "elseif";

      case Token::ElseWhen:
        return "elsewhen";

      case Token::Encapsulated:
        return "encapsulated";

      case Token::End:
        return "end";

      case Token::Enumeration:
        return "enumeration";

      case Token::Equation:
        return "equation";

      case Token::Expandable:
        return "expandable";

      case Token::Extends:
        return "extends";

      case Token::External:
        return "external";

      case Token::False:
        return "false";

      case Token::Final:
        return "final";

      case Token::Flow:
        return "flow";

      case Token::For:
        return "for";

      case Token::Function:
        return "function";

      case Token::If:
        return "if";

      case Token::Import:
        return "import";

      case Token::Impure:
        return "impure";

      case Token::In:
        return "in";

      case Token::Initial:
        return "initial";

      case Token::Inner:
        return "inner";

      case Token::Input:
        return "input";

      case Token::Loop:
        return "loop";

      case Token::Model:
        return "model";

      case Token::Not:
        return "not";

      case Token::Operator:
        return "operator";

      case Token::Or:
        return "or";

      case Token::Outer:
        return "outer";

      case Token::Output:
        return "output";

      case Token::Package:
        return "package";

      case Token::Parameter:
        return "parameter";

      case Token::Partial:
        return "partial";

      case Token::Protected:
        return "protected";

      case Token::Public:
        return "public";

      case Token::Pure:
        return "pure";

      case Token::Record:
        return "record";

      case Token::Redeclare:
        return "redeclare";

      case Token::Replaceable:
        return "replaceable";

      case Token::Return:
        return "return";

      case Token::Stream:
        return "stream";

      case Token::Then:
        return "then";

      case Token::True:
        return "true";

      case Token::Type:
        return "type";

      case Token::When:
        return "when";

      case Token::While:
        return "while";

      case Token::Within:
        return "within";

      case Token::Plus:
        return "+";

      case Token::PlusEW:
        return ".+";

      case Token::Minus:
        return "-";

      case Token::MinusEW:
        return ".-";

      case Token::Product:
        return "*";

      case Token::ProductEW:
        return ".*";

      case Token::Division:
        return "/";

      case Token::DivisionEW:
        return "./";

      case Token::Pow:
        return "^";

      case Token::PowEW:
        return ".^";

      case Token::Dot:
        return ".";

      case Token::Equal:
        return "==";

      case Token::NotEqual:
        return "<>";

      case Token::Less:
        return "<";

      case Token::LessEqual:
        return "<=";

      case Token::Greater:
        return ">";

      case Token::GreaterEqual:
        return ">=";

      case Token::Comma:
        return ",";

      case Token::Semicolon:
        return ";";

      case Token::Colon:
        return ":";

      case Token::LPar:
        return "(";

      case Token::RPar:
        return ")";

      case Token::LSquare:
        return "[";

      case Token::RSquare:
        return "]";

      case Token::LCurly:
        return "{";

      case Token::RCurly:
        return "}";

      case Token::EqualityOperator:
        return "=";

      case Token::AssignmentOperator:
        return ":=";
    }

    llvm_unreachable("Unknown token");
    return "unknown";
  }

  std::ostream& operator<<(std::ostream& os, const Token& obj)
  {
    return os << toString(obj);
  }

  llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Token& obj)
  {
    return os << toString(obj);
  }
}
