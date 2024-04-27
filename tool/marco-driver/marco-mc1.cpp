#include "marco/Frontend/CompilerInstance.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TargetSelect.h"

using namespace marco::frontend;

extern int mc1_main(llvm::ArrayRef<const char*> argv, const char* argv0)
{
  // Create the compiler instance.
  auto instance = std::make_unique<CompilerInstance>();

  // Initialize targets first, so that --version shows the registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  // Create the diagnostics engine.
  instance->createDiagnostics();

  // Parse the arguments.
  bool success = CompilerInvocation::createFromArgs(
      instance->getInvocation(), argv, instance->getDiagnostics());

  if (!success) {
    return 1;
  }

  // Execute the frontend actions.
  success = executeCompilerInvocation(instance.get());

  // Delete output files to free Compiler Instance.
  instance->clearOutputFiles(false);

  return !success;
}
