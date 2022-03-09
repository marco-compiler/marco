#include "llvm/ADT/ArrayRef.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/Support/TargetSelect.h"
#include "marco/frontend/CompilerInstance.h"
#include "marco/frontend/TextDiagnosticBuffer.h"
#include "marco/frontendTool/Utils.h"

using namespace marco::frontend;

extern int mc1_main(llvm::ArrayRef<const char*> argv)
{
  // Create CompilerInstance
  std::unique_ptr<CompilerInstance> instance(new CompilerInstance());

  // Create diagnostics engine for the frontend driver
  instance->createDiagnostics();

  // We will buffer diagnostics from argument parsing so that we can output
  // them using a well-formed diagnostic object.
  TextDiagnosticBuffer* diagnosticBuffer = new TextDiagnosticBuffer();

  // Create CompilerInvocation - use a dedicated instance of DiagnosticsEngine
  // for parsing the arguments
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagnosticID(new clang::DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnosticOptions = new clang::DiagnosticOptions();
  clang::DiagnosticsEngine diagnosticEngine(diagnosticID, &*diagnosticOptions, diagnosticBuffer);

  // Initialize targets first, so that --version shows the registered targets
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  bool success = CompilerInvocation::createFromArgs(instance->invocation(), argv, diagnosticEngine);

  diagnosticBuffer->FlushDiagnostics(instance->getDiagnostics());

  if (!success) {
    return 1;
  }

  // Execute the frontend actions
  success = executeCompilerInvocation(instance.get());

  // Delete output files to free Compiler Instance
  instance->clearOutputFiles(false);

  return !success;
}
