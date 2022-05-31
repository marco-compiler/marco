#include "marco/Parser/Message.h"
#include "marco/Diagnostic/Printer.h"

using namespace ::marco;
using namespace ::marco::diagnostic;
using namespace ::marco::parser;

namespace marco::parser
{
  UnexpectedTokenMessage::UnexpectedTokenMessage(const char* source, SourceRange location, Token actual, Token expected)
    : SourceMessage(source, std::move(location)),
      actual(actual),
      expected(expected)
  {
  }

  void UnexpectedTokenMessage::print(PrinterInstance* printer) const
  {
    auto& os = printer->getOutputStream();

    auto highlightSourceFn = [&](llvm::raw_ostream& os) {
      printer->setColor(os, printer->diagnosticLevelColor());
    };

    printFileNameAndPosition(os);
    highlightSourceFn(os);
    printDiagnosticLevel(os, printer->diagnosticLevel());
    printer->resetColor(os);
    os << ": ";

    os << "unexpected token: found '";

    printer->setBold(os);
    os << actual;
    printer->unsetBold(os);

    os << "' instead of '";

    printer->setBold(os);
    os << expected;
    printer->unsetBold(os);

    os << "'";
    os << "\n";

    printLines(os, highlightSourceFn);
  }

  UnexpectedIdentifierMessage::UnexpectedIdentifierMessage(const char* source, SourceRange location, std::string actual, std::string expected)
    : SourceMessage(source, std::move(location)),
      actual(std::move(actual)),
      expected(std::move(expected))
  {
  }

  void UnexpectedIdentifierMessage::print(diagnostic::PrinterInstance* printer) const
  {
    auto& os = printer->getOutputStream();

    auto highlightSourceFn = [&](llvm::raw_ostream& os) {
      printer->setColor(os, printer->diagnosticLevelColor());
    };

    printFileNameAndPosition(os);
    highlightSourceFn(os);
    printDiagnosticLevel(os, printer->diagnosticLevel());
    printer->resetColor(os);
    os << ": ";

    os << "unexpected identifier: found '";

    printer->setBold(os);
    os << actual;
    printer->unsetBold(os);

    os << "' instead of '";

    printer->setBold(os);
    os << expected;
    printer->unsetBold(os);

    os << "'";
    os << "\n";

    printLines(os, highlightSourceFn);
  }
}
