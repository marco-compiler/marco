#include "marco/Parser/Message.h"
#include "marco/Diagnostic/Printer.h"

using namespace ::marco;
using namespace ::marco::diagnostic;
using namespace ::marco::parser;

namespace marco::parser
{
  UnexpectedTokenMessage::UnexpectedTokenMessage(SourceRange location, Token actual, Token expected)
    : SourceMessage(std::move(location)),
      actual(std::move(actual)),
      expected(std::move(expected))
  {
  }

  void UnexpectedTokenMessage::print(PrinterInstance* printer) const
  {
    // TODO
  }

  UnexpectedIdentifierMessage::UnexpectedIdentifierMessage(SourceRange location, std::string actual, std::string expected)
    : SourceMessage(std::move(location)),
      actual(std::move(actual)),
      expected(std::move(expected))
  {
    // TODO
  }

  void UnexpectedIdentifierMessage::print(diagnostic::PrinterInstance* printer) const
  {
    // TODO
  }
}
