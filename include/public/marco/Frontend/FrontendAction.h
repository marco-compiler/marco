#ifndef MARCO_FRONTEND_FRONTENDACTION_H
#define MARCO_FRONTEND_FRONTENDACTION_H

#include "marco/Frontend/FrontendOptions.h"
#include "mlir/Pass/Pass.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"
#include <memory>

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

      const CompilerInstance& instance() const
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
      llvm::DataLayout getDataLayout();
      std::string getDataLayoutString();

      std::unique_ptr<mlir::Pass> createReadOnlyVariablesPropagationPass();
      std::unique_ptr<mlir::Pass> createAutomaticDifferentiationPass();
      std::unique_ptr<mlir::Pass> createModelLegalizationPass();
      std::unique_ptr<mlir::Pass> createMatchingPass();
      std::unique_ptr<mlir::Pass> createVariablesPromotionPass();
      std::unique_ptr<mlir::Pass> createCyclesSolvingPass();
      std::unique_ptr<mlir::Pass> createSchedulingPass();

      std::unique_ptr<mlir::Pass> createEulerForwardPass();
      std::unique_ptr<mlir::Pass> createIDAPass();

      std::unique_ptr<mlir::Pass> createFunctionScalarizationPass();
      std::unique_ptr<mlir::Pass> createExplicitCastInsertionPass();
      std::unique_ptr<mlir::Pass> createModelicaToCFConversionPass();
      std::unique_ptr<mlir::Pass> createModelicaToVectorConversionPass();
      std::unique_ptr<mlir::Pass> createModelicaToArithConversionPass();
      std::unique_ptr<mlir::Pass> createModelicaToFuncConversionPass();
      std::unique_ptr<mlir::Pass> createModelicaToMemRefConversionPass();
      std::unique_ptr<mlir::Pass> createModelicaToLLVMConversionPass();

      std::unique_ptr<mlir::Pass> createIDAToFuncConversionPass();
      std::unique_ptr<mlir::Pass> createIDAToLLVMConversionPass();

      std::unique_ptr<mlir::Pass> createKINSOLToLLVMConversionPass();

      std::unique_ptr<mlir::Pass> createSimulationToFuncConversionPass();

      std::unique_ptr<mlir::Pass> createFuncToLLVMConversionPass(
          bool useBarePtrCallConv);

      std::unique_ptr<mlir::Pass> createArithToLLVMConversionPass();
      std::unique_ptr<mlir::Pass> createMemRefToLLVMConversionPass();
      std::unique_ptr<mlir::Pass> createVectorToLLVMConversionPass();
      std::unique_ptr<mlir::Pass> createVectorToSCFConversionPass();

    private:
      CompilerInstance* instance_;
  };
}

#endif // MARCO_FRONTEND_FRONTENDACTION_H
