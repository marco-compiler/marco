#ifndef MARCO_VARIABLEFILTER_ERROR_H
#define MARCO_VARIABLEFILTER_ERROR_H

#include "marco/VariableFilter/Token.h"
#include "marco/Diagnostic/Location.h"
#include "marco/Diagnostic/LogMessage.h"

namespace marco::vf
{
  class UnexpectedTokenMessage : public marco::diagnostic::SourceMessage
  {
    public:
      UnexpectedTokenMessage(SourceRange location, Token actual, Token expected);

      void print(diagnostic::PrinterInstance* printer) const override;

    private:
      Token actual;
      Token expected;
  };

  class EmptyRegexMessage : public marco::diagnostic::SourceMessage
  {
    public:
      EmptyRegexMessage(SourceRange location);

      void print(diagnostic::PrinterInstance* printer) const override;
  };

  class InvalidRegexMessage : public marco::diagnostic::SourceMessage
  {
    public:
      InvalidRegexMessage(SourceRange location);

      void print(diagnostic::PrinterInstance* printer) const override;
  };
}

#endif // MARCO_VARIABLEFILTER_ERROR_H
