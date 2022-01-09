#ifndef MARCO_FRONTEND_FRONTENDACTION_H
#define MARCO_FRONTEND_FRONTENDACTION_H

#include <llvm/Support/Error.h>

#include "FrontendOptions.h"

namespace marco::frontend
{
  class CompilerInstance;

  /// Abstract base class for the actions which can be performed by the frontend.
  class FrontendAction
  {
    public:
      FrontendAction() : instance_(nullptr)
      {
      }

      virtual ~FrontendAction() = default;

      CompilerInstance& instance()
      {
        assert(instance_ != nullptr && "Compiler instance not registered");
        return *instance_;
      }

      void setCompilerInstance(CompilerInstance* value)
      {
        instance_ = value;
      }

      virtual bool beginAction();

      virtual void execute() = 0;

    protected:
      bool runFlattening();
      bool runParse();
      bool runFrontendPasses();
      bool runASTConversion();
      bool runDialectConversion();
      bool runLLVMIRGeneration();

    private:
      CompilerInstance* instance_;
  };
}

#endif // MARCO_FRONTEND_FRONTENDACTION_H
