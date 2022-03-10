#ifndef MARCO_FRONTEND_TEXTDIAGNOSTICPRINTER_H
#define MARCO_FRONTEND_TEXTDIAGNOSTICPRINTER_H

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/raw_ostream.h"

namespace marco::frontend
{
  class TextDiagnosticPrinter : public clang::DiagnosticConsumer
  {
      llvm::raw_ostream& os_;
      llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts_;

      /// A string to prefix to error messages.
      std::string prefix_;

    public:
      TextDiagnosticPrinter(llvm::raw_ostream& os, clang::DiagnosticOptions* diags);

      ~TextDiagnosticPrinter() override;

      /// Set the diagnostic printer prefix string, which will be printed at the
      /// start of any diagnostics. If empty, no prefix string is used.
      void set_prefix(std::string value) { prefix_ = std::move(value); }

      void HandleDiagnostic(clang::DiagnosticsEngine::Level level, const clang::Diagnostic& info) override;
  };
}

#endif // MARCO_FRONTEND_TEXTDIAGNOSTICPRINTER_H
