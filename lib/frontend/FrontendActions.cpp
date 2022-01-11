#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
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

  bool CodegenAction::beginAction()
  {
    return runFlattening() && runParse() && runFrontendPasses() && runASTConversion() && runDialectConversion() && runLLVMIRGeneration();
  }

  void EmitLLVMIRAction::execute()
  {
    llvm::outs() << instance().getLLVMModule();
  }

  void EmitObjectAction::execute()
  {
    CompilerInstance& ci = instance();

    auto targetTriple = llvm::sys::getDefaultTargetTriple();
    ci.getLLVMModule().setTargetTriple(targetTriple);

    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);

    // Print an error and exit if we couldn't find the requested target.
    // This generally occurs if we've forgotten to initialise the
    // TargetRegistry or we have a bogus target triple.

    if (!target) {
      llvm::errs() << error;
      return;
    }

    auto cpu = "generic";
    auto features = "";

    llvm::TargetOptions opt;
    auto RM = llvm::Optional<llvm::Reloc::Model>();
    auto targetMachine = target->createTargetMachine(targetTriple, cpu, features, opt, RM);

    ci.getLLVMModule().setDataLayout(targetMachine->createDataLayout());
    ci.getLLVMModule().setTargetTriple(targetTriple);

    auto os = ci.createDefaultOutputFile(true, ci.getFrontendOptions().outputFile, "o");

    llvm::legacy::PassManager pass;
    auto fileType = llvm::CGFT_ObjectFile;

    if (targetMachine->addPassesToEmitFile(pass, *os, nullptr, fileType)) {
      llvm::errs() << "TargetMachine can't emit a file of this type";
      return;
    }

    pass.run(ci.getLLVMModule());
  }
}
