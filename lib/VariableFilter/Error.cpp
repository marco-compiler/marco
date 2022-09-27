#include "marco/VariableFilter/Error.h"
#include "marco/Diagnostic/Printer.h"

using namespace ::marco;
using namespace ::marco::vf;

namespace marco::vf
{
  UnexpectedTokenMessage::UnexpectedTokenMessage(SourceRange location, Token actual, Token expected)
      : SourceMessage(std::move(location)), actual(actual), expected(expected)
  {
  }

  void UnexpectedTokenMessage::print(diagnostic::PrinterInstance* printer) const
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

    os << "[variables filter] unexpected token: found '";

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

  EmptyRegexMessage::EmptyRegexMessage(SourceRange location)
      : SourceMessage(std::move(location))
  {
  }

  void EmptyRegexMessage::print(diagnostic::PrinterInstance* printer) const
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

    os << "[variables filter] empty regex";
    os << "\n";

    printLines(os, highlightSourceFn);
  }

  InvalidRegexMessage::InvalidRegexMessage(SourceRange location)
      : SourceMessage(std::move(location))
  {
  }

  void InvalidRegexMessage::print(diagnostic::PrinterInstance* printer) const
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

    os << "[variables filter] invalid regex";
    os << "\n";

    printLines(os, highlightSourceFn);
  }
}
