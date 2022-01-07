#ifndef MARCO_FRONTEND_FRONTENDACTIONS_H
#define MARCO_FRONTEND_FRONTENDACTIONS_H

#include "FrontendAction.h"

namespace marco::frontend
{
  class InitOnlyAction : public FrontendAction
  {
      void execute() override;
  };

  class EmitASTAction : public FrontendAction
  {
      void execute() override;
  };

  class EmitModelicaDialectAction : public FrontendAction
  {
      void execute() override;
  };

  class EmitLLVMDialectAction : public FrontendAction
  {
      void execute() override;
  };

  class EmitLLVMIRAction : public FrontendAction
  {
      void execute() override;
  };

  class EmitBitcodeAction : public FrontendAction
  {
      void execute() override;
  };
}

#endif // MARCO_FRONTEND_FRONTENDACTIONS_H
