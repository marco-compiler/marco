#ifndef MARCO_FRONTEND_FRONTENDACTIONS_H
#define MARCO_FRONTEND_FRONTENDACTIONS_H

#include "marco/AST/AST.h"
#include "marco/Frontend/FrontendAction.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Target/TargetMachine.h"

namespace marco::frontend
{
  //===-------------------------------------------------------------------===//
  // Custom actions
  //===-------------------------------------------------------------------===//

  class InitOnlyAction : public FrontendAction
  {
    public:
      void executeAction() override;

    private:
      void printCategory(
        llvm::raw_ostream& os, llvm::StringRef category) const;

      void printOption(
          llvm::raw_ostream& os, llvm::StringRef name, llvm::StringRef value);

      void printOption(
          llvm::raw_ostream& os, llvm::StringRef name, bool value);

      void printOption(
          llvm::raw_ostream& os, llvm::StringRef name, long value);

      void printOption(
          llvm::raw_ostream& os, llvm::StringRef name, double value);
  };

  //===-------------------------------------------------------------------===//
  // Preprocessing actions
  //===-------------------------------------------------------------------===//

  class PreprocessingAction : public FrontendAction
  {
    public:
      bool beginSourceFileAction() override;

    protected:
      std::string flattened;
  };

  class EmitFlattenedAction : public PreprocessingAction
  {
    public:
      void executeAction() override;
  };

  //===-------------------------------------------------------------------===//
  // AST actions
  //===-------------------------------------------------------------------===//

  enum class ASTActionKind
  {
    Parse
  };

  class ASTAction : public PreprocessingAction
  {
    protected:
      explicit ASTAction(ASTActionKind action);

    public:
      bool beginSourceFileAction() override;

    private:
      ASTActionKind action;

    protected:
      std::unique_ptr<ast::ASTNode> ast;
  };

  class EmitASTAction : public ASTAction
  {
    public:
      EmitASTAction();

      void executeAction() override;
  };

  //===-------------------------------------------------------------------===//
  // CodeGen actions
  //===-------------------------------------------------------------------===//

  enum class CodeGenActionKind
  {
    GenerateMLIR,
    GenerateLLVMIR
  };

  class CodeGenAction : public ASTAction
  {
    protected:
      explicit CodeGenAction(CodeGenActionKind action);

    public:
      ~CodeGenAction() override;

      bool beginSourceFileAction() override;

    protected:
      /// Set up the LLVM's TargetMachine.
      bool setUpTargetMachine();

      /// Get the data layout of the machine for which the code is being
      /// compiled.
      llvm::DataLayout getDataLayout() const;

      /// @name MLIR
      /// {

      /// Register the MLIR dialects that can be encountered while parsing.
      void registerMLIRDialects();

      /// Load the MLIR dialects used during the compilation.
      void loadMLIRDialects();

      /// Parse MLIR from the source file if in MLIR format, or generate it
      /// according to the source file type.
      bool generateMLIR();

      /// Set the target triple inside the MLIR module.
      void setMLIRModuleTargetTriple();

      /// Set the data layout inside the MLIR module.
      void setMLIRModuleDataLayout();

      /// Given a pass manager, append the passes to convert the Modelica
      /// dialect to the LLVM dialect
      void createModelicaToLLVMPassPipeline(mlir::PassManager& pm);

      /// }
      /// @name Modelica passes.
      /// {

      std::unique_ptr<mlir::Pass> createMLIRFunctionScalarizationPass();
      std::unique_ptr<mlir::Pass> createMLIRReadOnlyVariablesPropagationPass();

      std::unique_ptr<mlir::Pass> createMLIRCyclesSolvingPass();

      std::unique_ptr<mlir::Pass> createMLIREulerForwardPass();
      std::unique_ptr<mlir::Pass> createMLIRIDAPass();

      /// }
      /// @name Conversion passes.
      /// {

      std::unique_ptr<mlir::Pass> createMLIRModelicaToArithConversionPass();
      std::unique_ptr<mlir::Pass> createMLIRModelicaToCFConversionPass();
      std::unique_ptr<mlir::Pass> createMLIRModelicaToFuncConversionPass();
      std::unique_ptr<mlir::Pass> createMLIRModelicaToLLVMConversionPass();
      std::unique_ptr<mlir::Pass> createMLIRModelicaToMemRefConversionPass();
      std::unique_ptr<mlir::Pass> createMLIRModelicaToVectorConversionPass();

      std::unique_ptr<mlir::Pass> createMLIRIDAToFuncConversionPass();
      std::unique_ptr<mlir::Pass> createMLIRIDAToLLVMConversionPass();

      std::unique_ptr<mlir::Pass> createMLIRKINSOLToLLVMConversionPass();

      std::unique_ptr<mlir::Pass> createMLIRSimulationToFuncConversionPass();

      /// }
      /// @name MLIR built-in passes.
      /// {

      std::unique_ptr<mlir::Pass> createMLIRArithToLLVMConversionPass();

      std::unique_ptr<mlir::Pass>
      createMLIRFuncToLLVMConversionPass(bool useBarePtrCallConv);

      std::unique_ptr<mlir::Pass> createMLIRMemRefToLLVMConversionPass();
      std::unique_ptr<mlir::Pass> createMLIRVectorToLLVMConversionPass();
      std::unique_ptr<mlir::Pass> createMLIRVectorToSCFConversionPass();

      /// }
      /// @name LLVM-IR
      /// {

      /// Register the translations needed to emit LLVM-IR.
      void registerMLIRToLLVMIRTranslations();

      /// Parse LLVM-IR from the source file if in LLVM-IR format, or generate
      /// it according to the source file type.
      bool generateLLVMIR();

      /// Set the target triple inside the LLVM module.
      void setLLVMModuleTargetTriple();

      /// Set the data layout inside the LLVM module.
      void setLLVMModuleDataLayout();

      /// Run the optimization (aka middle-end) pipeline on the LLVM module
      /// associated with this action.
      void runOptimizationPipeline();

      /// }

    private:
      CodeGenActionKind action;

    protected:
      mlir::DialectRegistry mlirDialectRegistry;
      std::unique_ptr<mlir::MLIRContext> mlirContext;
      std::unique_ptr<mlir::ModuleOp> mlirModule;

      std::unique_ptr<llvm::LLVMContext> llvmContext;
      std::unique_ptr<llvm::Module> llvmModule;

      std::unique_ptr<llvm::TargetMachine> targetMachine;
  };

  class EmitMLIRAction : public CodeGenAction
  {
    public:
      EmitMLIRAction();

      void executeAction() override;
  };

  class EmitLLVMIRAction : public CodeGenAction
  {
    public:
      EmitLLVMIRAction();

      void executeAction() override;
  };

  class EmitBitcodeAction : public CodeGenAction
  {
    public:
      EmitBitcodeAction();

      void executeAction() override;
  };

  class EmitAssemblyAction : public CodeGenAction
  {
    public:
      EmitAssemblyAction();

      void executeAction() override;
  };

  class EmitObjAction : public CodeGenAction
  {
    public:
      EmitObjAction();

      void executeAction() override;
  };
}

#endif // MARCO_FRONTEND_FRONTENDACTIONS_H
