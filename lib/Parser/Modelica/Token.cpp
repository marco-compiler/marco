#include "marco/Parser/Modelica/Token.h"

using namespace ::marco::parser::modelica;

namespace marco::parser::modelica {
llvm::StringRef toString(TokenKind kind) {
  switch (kind) {
  case TokenKind::EndOfFile:
    return "end of file";
  case TokenKind::Error:
    return "error";
  case TokenKind::Identifier:
    return "identifier";
  case TokenKind::Integer:
    return "integer";
  case TokenKind::Real:
    return "real";
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
  case TokenKind::Time:
    return "time";
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
  case TokenKind::Star:
    return "*";
  case TokenKind::StarEW:
    return ".*";
  case TokenKind::Slash:
    return "/";
  case TokenKind::SlashEW:
    return "./";
  case TokenKind::Caret:
    return "^";
  case TokenKind::CaretEW:
    return ".^";
  case TokenKind::Equal:
    return "=";
  case TokenKind::EqualEqual:
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
  case TokenKind::Assignment:
    return ":=";
  case TokenKind::Comma:
    return ",";
  case TokenKind::Semicolon:
    return ";";
  case TokenKind::Colon:
    return ":";
  case TokenKind::Dot:
    return ".";
  case TokenKind::LParen:
    return "(";
  case TokenKind::RParen:
    return ")";
  case TokenKind::LSquare:
    return "[";
  case TokenKind::RSquare:
    return "]";
  case TokenKind::LCurly:
    return "{";
  case TokenKind::RCurly:
    return "}";
  }

  llvm_unreachable("Unknown token kind");
  return "";
}

std::string Token::describe() const {
  switch (kind) {
  case TokenKind::Identifier:
    return "identifier '" + text + "'";
  case TokenKind::Integer:
    return "integer " + std::to_string(integerValue);
  case TokenKind::Real:
    return "real " + std::to_string(realValue);
  case TokenKind::String:
    return "string \"" + text + "\"";
  case TokenKind::EndOfFile:
    return "end of file";
  default:
    return "'" + toString(kind).str() + "'";
  }
}
} // namespace marco::parser::modelica
