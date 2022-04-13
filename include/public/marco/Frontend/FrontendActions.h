#ifndef MARCO_FRONTEND_FRONTENDACTIONS_H
#define MARCO_FRONTEND_FRONTENDACTIONS_H

#include "marco/Frontend/FrontendAction.h"

namespace marco::frontend
{
  class InitOnlyAction : public FrontendAction
  {
    public:
      void execute() override;

    private:
      llvm::raw_ostream& getOutputStream() const;

      void printCategory(llvm::raw_ostream& os, llvm::StringRef category) const;

      void printOption(llvm::raw_ostream& os, llvm::StringRef name, llvm::StringRef value);
      void printOption(llvm::raw_ostream& os, llvm::StringRef name, bool value);
      void printOption(llvm::raw_ostream& os, llvm::StringRef name, long value);
      void printOption(llvm::raw_ostream& os, llvm::StringRef name, double value);
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

  class CompileAction : public CodegenAction
  {
    protected:
      void compileAndEmitFile(llvm::CodeGenFileType fileType, llvm::raw_pwrite_stream& os);
  };

  class EmitAssemblyAction : public CompileAction
  {
    public:
      void execute() override;
  };

  class EmitObjectAction : public CompileAction
  {
    public:
      void execute() override;
  };
}

#endif // MARCO_FRONTEND_FRONTENDACTIONS_H
