#include "marco/Frontend/CompilerInstance.h"
#include "marco/Frontend/FrontendActions.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/Host.h"

using namespace ::marco;
using namespace ::marco::diagnostic;
using namespace ::marco::frontend;

namespace marco::frontend
{
  void InitOnlyAction::execute()
  {
    CompilerInstance& ci = this->instance();

    ci.getDiagnostics().emitWarning<GenericStringMessage>(
        "Use '-init-only' for testing purposes only");

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
    printOption(os, "Target triple", codegenOptions.target);
    printOption(os, "Target cpu", codegenOptions.cpu);
    os << "\n";

    const auto& simulationOptions = ci.getSimulationOptions();
    printCategory(os, "Simulation");
    printOption(os, "Model", simulationOptions.modelName);
    printOption(os, "Start time", simulationOptions.startTime);
    printOption(os, "End time", simulationOptions.endTime);
    printOption(os, "Time step", simulationOptions.timeStep);

    std::string solver;

    if (simulationOptions.solver.getKind() == codegen::Solver::Kind::forwardEuler) {
      solver = "Forward Euler";
    } else if (simulationOptions.solver.getKind() == codegen::Solver::Kind::ida) {
      solver = "IDA";
    }

    printOption(os, "Solver", solver);
    os << "\n";

    // IDA
    printCategory(os, "IDA");
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

    if (auto targetMachine = ci.getTargetMachine()) {
      // Compile the module
      llvm::legacy::PassManager passManager;

      if (targetMachine->addPassesToEmitFile(passManager, os, nullptr, fileType)) {
        ci.getDiagnostics().emitFatalError<GenericStringMessage>(
            "TargetMachine can't emit a file of this type");

        return;
      }

      passManager.run(ci.getLLVMModule());
      os.flush();
    }
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
