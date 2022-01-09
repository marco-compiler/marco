#include <llvm/Bitcode/BitcodeWriter.h>
#include <marco/frontend/CompilerInstance.h>
#include <marco/frontend/FrontendActions.h>

namespace marco::frontend
{
  void InitOnlyAction::execute()
  {
    CompilerInstance& ci = this->instance();

    unsigned int DiagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Use '-init-only' for testing purposes only");

    ci.getDiagnostics().Report(DiagID);
  }

  bool EmitFlattenedAction::beginAction()
  {
    return runFlattening();
  }

  void EmitFlattenedAction::execute()
  {
    llvm::outs() << instance().getFlattened() << "\n";
  }

  bool EmitASTAction::beginAction()
  {
    return runFlattening() && runParse();
  }

  void EmitASTAction::execute()
  {
    instance().getAST()->dump(llvm::outs());
  }

  bool EmitModelicaDialectAction::beginAction()
  {
    return runFlattening() && runParse() && runFrontendPasses() && runASTConversion();
  }

  void EmitModelicaDialectAction::execute()
  {
    instance().getMLIRModule().print(llvm::outs());
  }

  bool EmitLLVMDialectAction::beginAction()
  {
    return runFlattening() && runParse() && runFrontendPasses() && runASTConversion() && runDialectConversion();
  }

  void EmitLLVMDialectAction::execute()
  {
    instance().getMLIRModule().print(llvm::outs());
  }

  bool EmitLLVMIRAction::beginAction()
  {
    return runFlattening() && runParse() && runFrontendPasses() && runASTConversion() && runDialectConversion() && runLLVMIRGeneration();
  }

  void EmitLLVMIRAction::execute()
  {
    llvm::outs() << instance().getLLVMModule();
  }

  bool EmitBitcodeAction::beginAction()
  {
    return runFlattening() && runParse() && runFrontendPasses() && runASTConversion() && runDialectConversion() && runLLVMIRGeneration();
  }

  void EmitBitcodeAction::execute()
  {
    CompilerInstance& ci = instance();
    auto os = ci.createDefaultOutputFile(true, ci.getFrontendOptions().outputFile, "bc");
    llvm::WriteBitcodeToFile(ci.getLLVMModule(), *os);
  }
}
