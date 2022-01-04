#include <clang/Basic/DiagnosticOptions.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <marco/frontend/TextDiagnostic.h>
#include <marco/frontend/TextDiagnosticPrinter.h>

namespace marco::frontend
{
  TextDiagnosticPrinter::TextDiagnosticPrinter(
      llvm::raw_ostream& os, clang::DiagnosticOptions* diags)
      : os_(os), diagOpts_(diags)
  {
  }

  TextDiagnosticPrinter::~TextDiagnosticPrinter() = default;

  void TextDiagnosticPrinter::HandleDiagnostic(
      clang::DiagnosticsEngine::Level level, const clang::Diagnostic& info)
  {
    // Default implementation (Warnings/errors count).
    DiagnosticConsumer::HandleDiagnostic(level, info);

    // Render the diagnostic message into a temporary buffer eagerly. We'll use
    // this later as we print out the diagnostic to the terminal.
    llvm::SmallString<100> outStr;
    info.FormatDiagnostic(outStr);

    llvm::raw_svector_ostream DiagMessageStream(outStr);

    if (!prefix_.empty()) {
      os_ << prefix_ << ": ";
    }

    // We only emit diagnostics in contexts that lack valid source locations.
    assert(!info.getLocation().isValid() &&
        "Diagnostics with valid source location are not supported");

    TextDiagnostic::PrintDiagnosticLevel(os_, level, diagOpts_->ShowColors);

    TextDiagnostic::PrintDiagnosticMessage(
        os_,
        level == clang::DiagnosticsEngine::Note,
        DiagMessageStream.str(), diagOpts_->ShowColors);

    os_.flush();
    return;
  }
}
