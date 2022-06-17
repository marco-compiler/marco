#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "marco/Frontend/CompilerInstance.h"
#include "marco/Frontend/FrontendActions.h"

namespace marco::frontend
{
  void InitOnlyAction::execute()
  {
    CompilerInstance& ci = this->instance();

    unsigned int DiagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Use '-init-only' for testing purposes only");

    ci.getDiagnostics().Report(DiagID);

    auto& os = llvm::outs();

    const auto& codegenOptions = ci.getCodegenOptions();
    printCategory(os, "Code generation");
    printOption(os, "Time optimization level", static_cast<long>(codegenOptions.optLevel.time));
    printOption(os, "Size optimization level", static_cast<long>(codegenOptions.optLevel.size));
    printOption(os, "Debug information", codegenOptions.debug);
    printOption(os, "Assertions", codegenOptions.assertions);
    printOption(os, "Inlining", codegenOptions.inlining);
    printOption(os, "Output arrays promotion", codegenOptions.outputArraysPromotion);
    printOption(os, "CSE", codegenOptions.cse);
    printOption(os, "OpenMP", codegenOptions.omp);
    printOption(os, "Main function generation", codegenOptions.generateMain);
    os << "\n";

    const auto& simulationOptions = ci.getSimulationOptions();
    printCategory(os, "Simulation");
    printOption(os, "Model", simulationOptions.modelName);
    printOption(os, "Start time", simulationOptions.startTime);
    printOption(os, "End time", simulationOptions.endTime);
    printOption(os, "Time step", simulationOptions.timeStep);

    std::string solver;

    if (simulationOptions.solver == codegen::Solver::forwardEuler) {
      solver = "Forward Euler";
    } else if (simulationOptions.solver == codegen::Solver::ida) {
      solver = "IDA";
    }

    printOption(os, "Solver", solver);
    os << "\n";

    // IDA
    printCategory(os, "IDA");
    printOption(os, "Relative tolerance", simulationOptions.ida.relativeTolerance);
    printOption(os, "Absolute tolerance", simulationOptions.ida.absoluteTolerance);
    printOption(os, "Equidistant time grid", simulationOptions.ida.equidistantTimeGrid);
    os << "\n";
  }

  void InitOnlyAction::printCategory(llvm::raw_ostream& os, llvm::StringRef category) const
  {
    os << "[" << category << "]\n";
  }

  void InitOnlyAction::printOption(llvm::raw_ostream& os, llvm::StringRef name, llvm::StringRef value)
  {
    os << " - " << name << ": " << value << "\n";
  }

  void InitOnlyAction::printOption(llvm::raw_ostream& os, llvm::StringRef name, bool value)
  {
    os << " - " << name << ": " << (value ? "true" : "false") << "\n";
  }

  void InitOnlyAction::printOption(llvm::raw_ostream& os, llvm::StringRef name, long value)
  {
    os << " - " << name << ": " << value << "\n";
  }

  void InitOnlyAction::printOption(llvm::raw_ostream& os, llvm::StringRef name, double value)
  {
    os << " - " << name << ": " << value << "\n";
  }

  bool EmitFlattenedAction::beginAction()
  {
    return runFlattening();
  }

  void EmitFlattenedAction::execute()
  {
    // Print the flattened source on the standard output
    llvm::outs() << instance().getFlattened() << "\n";
  }

  bool EmitASTAction::beginAction()
  {
    return runFlattening() && runParse();
  }

  void EmitASTAction::execute()
  {
    // Print the AST on the standard output
    instance().getAST()->dump(llvm::outs());
  }

  bool EmitFinalASTAction::beginAction()
  {
    return runFlattening() && runParse() && runFrontendPasses();
  }

  void EmitFinalASTAction::execute()
  {
    // Print the AST on the standard output
    instance().getAST()->dump(llvm::outs());
  }

  bool EmitModelicaDialectAction::beginAction()
  {
    return runFlattening() && runParse() && runFrontendPasses() && runASTConversion();
  }

  void EmitModelicaDialectAction::execute()
  {
    // Print the module on the standard output
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
    // Print the module on the standard output
    llvm::outs() << instance().getLLVMModule();
  }

  void CompileAction::compileAndEmitFile(llvm::CodeGenFileType fileType, llvm::raw_pwrite_stream& os)
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
      unsigned int diagId = ci.getDiagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error, "%s");
      ci.getDiagnostics().Report(diagId) << error;
      return;
    }

    std::string cpu = std::string(llvm::sys::getHostCPUName());
    auto features = "";

    llvm::TargetOptions opt;
    auto relocationModel = llvm::Optional<llvm::Reloc::Model>(llvm::Reloc::PIC_);
    auto targetMachine = target->createTargetMachine(targetTriple, cpu, features, opt, relocationModel);

    ci.getLLVMModule().setDataLayout(targetMachine->createDataLayout());

    llvm::legacy::PassManager passManager;

    if (targetMachine->addPassesToEmitFile(passManager, os, nullptr, fileType)) {
      unsigned int diagId = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "TargetMachine can't emit a file of this type");

      ci.getDiagnostics().Report(diagId);
      return;
    }

    passManager.run(ci.getLLVMModule());
    os.flush();
  }

  void EmitAssemblyAction::execute()
  {
    CompilerInstance& ci = instance();
    auto os = ci.createDefaultOutputFile(false, ci.getFrontendOptions().outputFile, "s");
    compileAndEmitFile(llvm::CGFT_AssemblyFile, *os);
  }

  void EmitObjectAction::execute()
  {
    CompilerInstance& ci = instance();
    auto os = ci.createDefaultOutputFile(true, ci.getFrontendOptions().outputFile, "o");
    compileAndEmitFile(llvm::CGFT_ObjectFile, *os);
  }
}
