#include "llvm/Support/CommandLine.h"
#include "marco/frontend/CompilerInstance.h"
#include "marco/frontend/FrontendActions.h"
#include "marco/frontend/Options.h"
#include "marco/frontendTool/Utils.h"

using namespace marco::frontend;

static std::unique_ptr<FrontendAction> createFrontendBaseAction(CompilerInstance& ci)
{
  ActionKind action = ci.getFrontendOptions().programAction;

  switch (action) {
    case InitOnly:
      return std::make_unique<InitOnlyAction>();

    case EmitFlattened:
      return std::make_unique<EmitFlattenedAction>();

    case EmitAST:
      return std::make_unique<EmitASTAction>();

    case EmitFinalAST:
      return std::make_unique<EmitFinalASTAction>();

    case EmitModelicaDialect:
      return std::make_unique<EmitModelicaDialectAction>();

    case EmitLLVMDialect:
      return std::make_unique<EmitLLVMDialectAction>();

    case EmitLLVMIR:
      return std::make_unique<EmitLLVMIRAction>();

    case EmitObject:
      return std::make_unique<EmitObjectAction>();

    default:
      break;
  }

  llvm_unreachable("Invalid program action");
  return 0;
}

namespace marco::frontend
{
  std::unique_ptr<FrontendAction> createFrontendAction(CompilerInstance& ci)
  {
    // Create the underlying action.
    std::unique_ptr<FrontendAction> act = createFrontendBaseAction(ci);

    if (!act) {
      return nullptr;
    }

    return act;
  }

  bool executeCompilerInvocation(CompilerInstance* ci)
  {
    // Honor --help
    if (ci->getFrontendOptions().showHelp) {
      getDriverOptTable().printHelp(
          llvm::outs(),
          "marco-driver -mc1 [options] input-files", "MLIR Modelica compiler",
          marco::frontend::options::MC1Option,
          llvm::opt::DriverFlag::HelpHidden,
          false);

      return true;
    }

    // Honor --version
    if (ci->getFrontendOptions().showVersion) {
      llvm::cl::PrintVersionMessage();
      return true;
    }

    // If there were errors in processing arguments, don't do anything else
    if (ci->getDiagnostics().hasErrorOccurred()) {
      return false;
    }

    // Create and execute the frontend action
    std::unique_ptr<FrontendAction> act(createFrontendAction(*ci));

    if (!act) {
      return false;
    }

    return ci->executeAction(*act);
  }
}
