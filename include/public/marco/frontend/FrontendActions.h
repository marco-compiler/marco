#ifndef MARCO_FRONTEND_FRONTENDACTIONS_H
#define MARCO_FRONTEND_FRONTENDACTIONS_H

#include "FrontendAction.h"

namespace marco::frontend
{
  class InitOnlyAction : public FrontendAction
  {
    public:
      void execute() override;
  };

  class EmitFlattenedAction : public FrontendAction
  {
    public:
      bool beginAction() override;
      void execute() override;
  };

  class EmitASTAction : public FrontendAction
  {
    public:
      bool beginAction() override;
      void execute() override;
  };

  class EmitModelicaDialectAction : public FrontendAction
  {
    public:
      bool beginAction() override;
      void execute() override;
  };

  class EmitLLVMDialectAction : public FrontendAction
  {
    public:
      bool beginAction() override;
      void execute() override;
  };

  class EmitLLVMIRAction : public FrontendAction
  {
    public:
      bool beginAction() override;
      void execute() override;
  };

  class EmitBitcodeAction : public FrontendAction
  {
    public:
      bool beginAction() override;
      void execute() override;
  };
}

#endif // MARCO_FRONTEND_FRONTENDACTIONS_H
