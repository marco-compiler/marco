#ifndef MARCO_FRONTEND_COMPILERINVOCATION_H
#define MARCO_FRONTEND_COMPILERINVOCATION_H

#include "marco/Diagnostic/Diagnostic.h"
#include "marco/Frontend/CodegenOptions.h"
#include "marco/Frontend/FrontendOptions.h"
#include "marco/Frontend/SimulationOptions.h"
#include "llvm/Option/ArgList.h"
#include <memory>

namespace marco::frontend
{
  class CompilerInvocation
  {
    public:
      /// Create a compiler invocation from a list of input options.
      static bool createFromArgs(
        CompilerInvocation& res,
        llvm::ArrayRef<const char*> commandLineArgs,
        diagnostic::DiagnosticEngine& diagnostics);

      FrontendOptions& getFrontendOptions();

      const FrontendOptions& getFrontendOptions() const;

      CodegenOptions& getCodeGenOptions();

      const CodegenOptions& getCodeGenOptions() const;

      SimulationOptions& getSimulationOptions();

      const SimulationOptions& getSimulationOptions() const;

    private:
      FrontendOptions frontendOptions;
      CodegenOptions codegenOptions;
      SimulationOptions simulationOptions;
  };
}

#endif // MARCO_FRONTEND_COMPILERINVOCATION_H
