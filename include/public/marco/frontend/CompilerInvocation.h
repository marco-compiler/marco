#ifndef MARCO_FRONTEND_COMPILERINVOCATION_H
#define MARCO_FRONTEND_COMPILERINVOCATION_H

#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <llvm/Option/ArgList.h>
#include <memory>

#include "FrontendOptions.h"

namespace marco::frontend
{
  /// Fill out Opts based on the options given in Args.
  ///
  /// When errors are encountered, return false and, if Diags is non-null,
  /// report the error(s).
  bool ParseDiagnosticArgs(
      clang::DiagnosticOptions& opts,
      llvm::opt::ArgList& args, bool defaultDiagColor = true);

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
      /// Options for the frontend driver
      FrontendOptions frontendOpts_;

      bool debugModuleDir_ = false;

    public:
      CompilerInvocation() = default;

      FrontendOptions& frontendOpts() { return frontendOpts_; }

      const FrontendOptions& frontendOpts() const { return frontendOpts_; }

      /// Create a compiler invocation from a list of input options.
      /// \returns true on success.
      /// \returns false if an error was encountered while parsing the arguments
      /// \param [out] res - The resulting invocation.
      static bool CreateFromArgs(
          CompilerInvocation& res,
          llvm::ArrayRef<const char*> commandLineArgs,
          clang::DiagnosticsEngine& diags);

      /// Set the Fortran options to predefined defaults.
      // TODO: We should map frontendOpts_ to parserOpts_ instead. For that, we
      // need to extend frontendOpts_ first. Next, we need to add the corresponding
      // compiler driver options in libclangDriver.
      void SetDefaultFortranOpts();

      /// Set the Fortran options to user-specified values.
      /// These values are found in the preprocessor options.
      void setFortranOpts();
  };
}

#endif // MARCO_FRONTEND_COMPILERINVOCATION_H
