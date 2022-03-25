#ifndef MARCO_FRONTEND_FRONTENDACTIONS_H
#define MARCO_FRONTEND_FRONTENDACTIONS_H

#include "marco/Frontend/FrontendAction.h"

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

  class EmitFinalASTAction : public FrontendAction
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

  class CodegenAction : public FrontendAction
  {
    public:
      bool beginAction() override;
  };

  class EmitLLVMIRAction : public CodegenAction
  {
    public:
      void execute() override;
  };

  class EmitObjectAction : public CodegenAction
  {
    public:
      void execute() override;
  };
}

#endif // MARCO_FRONTEND_FRONTENDACTIONS_H
