#ifndef MARCO_FRONTEND_COMPILERINVOCATION_H
#define MARCO_FRONTEND_COMPILERINVOCATION_H

#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <llvm/Option/ArgList.h>
#include <memory>

#include "CodegenOptions.h"
#include "DialectOptions.h"
#include "FrontendOptions.h"
#include "SimulationOptions.h"

namespace marco::frontend
{
  /// Fill out Opts based on the options given in Args.
  ///
  /// When errors are encountered, return false and, if Diags is non-null,
  /// report the error(s).
  /*
  bool ParseDiagnosticArgs(
      clang::DiagnosticOptions& opts,
      llvm::opt::ArgList& args, bool defaultDiagColor = true);
      */

  class CompilerInvocationBase
  {
    public:
      /// Options controlling the diagnostic engine.
      llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnosticOpts_;

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
      static bool CreateFromArgs(
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
