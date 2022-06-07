#ifndef MARCO_PARSER_ERROR_H
#define MARCO_PARSER_ERROR_H

#include "marco/Diagnostic/Diagnostic.h"
#include "marco/Parser/Token.h"
#include "llvm/ADT/StringRef.h"

namespace marco::diagnostic
{
  class PrinterInstance;
}

namespace marco::parser
{
  class UnexpectedTokenMessage : public diagnostic::SourceMessage
  {
    public:
      UnexpectedTokenMessage(SourceRange location, Token actual, Token expected);

      void print(diagnostic::PrinterInstance* printer) const override;

    private:
      Token actual;
      Token expected;
  };

  class UnexpectedIdentifierMessage : public diagnostic::SourceMessage
  {
    public:
      UnexpectedIdentifierMessage(SourceRange location, std::string actual, std::string expected);

      void print(diagnostic::PrinterInstance* printer) const override;

    private:
      std::string actual;
      std::string expected;
  };
}

#endif // MARCO_PARSER_ERROR_H
