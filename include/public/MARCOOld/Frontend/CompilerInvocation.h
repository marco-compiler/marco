#ifndef MARCO_FRONTEND_COMPILERINVOCATION_H
#define MARCO_FRONTEND_COMPILERINVOCATION_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/Option/ArgList.h"
#include "marco/Frontend/CodegenOptions.h"
#include "marco/Frontend/DialectOptions.h"
#include "marco/Frontend/FrontendOptions.h"
#include "marco/Frontend/SimulationOptions.h"
#include <memory>

namespace marco::frontend
{
  class CompilerInvocationBase
  {
    public:
      CompilerInvocationBase();

      CompilerInvocationBase(const CompilerInvocationBase& x);

      ~CompilerInvocationBase();

      clang::DiagnosticOptions& GetDiagnosticOpts()
      {
        return *diagnosticOpts_.get();
      }

      const clang::DiagnosticOptions& GetDiagnosticOpts() const
      {
        return *diagnosticOpts_.get();
      }

    private:
      /// Options controlling the diagnostic engine.
      llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnosticOpts_;
  };

  class CompilerInvocation : public CompilerInvocationBase
  {
    public:
      CompilerInvocation() = default;

      FrontendOptions& frontendOptions() { return frontendOptions_; }

      const FrontendOptions& frontendOptions() const { return frontendOptions_; }

      DialectOptions& dialectOptions() { return dialectOptions_; }

      const DialectOptions& dialectOptions() const { return dialectOptions_; }

      CodegenOptions& codegenOptions() { return codegenOptions_; }

      const CodegenOptions& codegenOptions() const { return codegenOptions_; }

      SimulationOptions& simulationOptions() { return simulationOptions_; }

      const SimulationOptions& simulationOptions() const { return simulationOptions_; }

      /// Create a compiler invocation from a list of input options.
      /// \returns true on success.
      /// \returns false if an error was encountered while parsing the arguments
      /// \param [out] res - The resulting invocation.
      static bool createFromArgs(
          CompilerInvocation& res,
          llvm::ArrayRef<const char*> commandLineArgs,
          clang::DiagnosticsEngine& diags);

    private:
      FrontendOptions frontendOptions_;
      DialectOptions dialectOptions_;
      CodegenOptions codegenOptions_;
      SimulationOptions simulationOptions_;
  };
}

#endif // MARCO_FRONTEND_COMPILERINVOCATION_H
