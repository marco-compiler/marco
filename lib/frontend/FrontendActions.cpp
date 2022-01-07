#include <marco/frontend/CompilerInstance.h>
#include <marco/frontend/FrontendActions.h>

namespace marco::frontend
{
  void InitOnlyAction::execute()
  {
    CompilerInstance& ci = this->instance();

    unsigned int DiagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Use '-init-only' for testing purposes only");

    ci.diagnostics().Report(DiagID);
  }

  void EmitASTAction::execute()
  {
    if (!runParse()) {
      return;
    }

    for (const auto& cls : instance().classes()) {
      cls->dump(llvm::outs());
    }
  }

  void EmitModelicaDialectAction::execute()
  {
    if (!runParse() || !runFrontendPasses() || !runASTConversion()) {
      return;
    }

    instance().mlirModule().dump();
  }

  void EmitLLVMDialectAction::execute()
  {
    if (!runParse() || !runFrontendPasses() || !runASTConversion() || !runDialectConversion()) {
      return;
    }

    instance().mlirModule().dump();
  }

  void EmitLLVMIRAction::execute()
  {
    if (!runParse() || !runFrontendPasses() || !runASTConversion() || !runDialectConversion() || !runLLVMIRGeneration()) {
      return;
    }

    instance().llvmModule().dump();
  }

  void EmitBitcodeAction::execute()
  {
    if (!runParse() || !runFrontendPasses() || !runASTConversion() || !runDialectConversion() || !runLLVMIRGeneration()) {
      return;
    }

    // TODO
  }
}
