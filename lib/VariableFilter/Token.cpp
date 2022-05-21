#include "marco/VariableFilter/Token.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco;
using namespace ::marco::vf;

namespace marco::vf
{
  std::string toString(Token token)
  {
    switch (token)
    {
      case Token::BeginOfFile:
        return "Begin";
      case Token::EndOfFile:
        return "EOF";
      case Token::Error:
        return "Error";
      case Token::None:
        return "None";
      case Token::Integer:
        return "Integer";
      case Token::Ident:
        return "Identifier";
      case Token::Regex:
        return "Regex";
      case Token::LPar:
        return "(";
      case Token::RPar:
        return ")";
      case Token::LSquare:
        return "[";
      case Token::RSquare:
        return "]";
      case Token::Comma:
        return ",";
      case Token::Semicolons:
        return ";";
      case Token::Colons:
        return ":";
      case Token::Dollar:
        return "$";
      case Token::DerKeyword:
        return "der";
    }

    return "[Unexpected]";
  }

  llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Token& obj)
  {
    return stream << toString(obj);
  }
}
