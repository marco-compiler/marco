#ifndef MARCO_VARIABLEFILTER_TOKEN_H
#define MARCO_VARIABLEFILTER_TOKEN_H

#include <string>

namespace llvm
{
  class raw_ostream;
}

namespace marco::vf
{
  enum class Token
  {
    // Control tokens
    BeginOfFile,
    EndOfFile,
    Error,
    None,

    // Placeholders
    Integer,
    Ident,
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

  std::string toString(Token token);

  llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Token& obj);
}

#endif // MARCO_VARIABLEFILTER_TOKEN_H
