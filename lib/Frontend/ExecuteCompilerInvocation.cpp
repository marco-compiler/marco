#include "marco/Options/Options.h"
#include "marco/Frontend/CompilerInstance.h"
#include "marco/Frontend/FrontendActions.h"
#include "llvm/Support/CommandLine.h"

using namespace ::marco::frontend;

static std::unique_ptr<FrontendAction> createFrontendBaseAction(
    CompilerInstance& ci)
{
  ActionKind action = ci.getFrontendOptions().programAction;

  switch (action) {
    case InitOnly:
      return std::make_unique<InitOnlyAction>();

    case EmitFlattened:
      return std::make_unique<EmitFlattenedAction>();

    case EmitAST:
      return std::make_unique<EmitASTAction>();

    case EmitMLIR:
      return std::make_unique<EmitMLIRAction>();

    case EmitLLVMIR:
      return std::make_unique<EmitLLVMIRAction>();

    case EmitLLVMBitcode:
      return std::make_unique<EmitBitcodeAction>();

    case EmitAssembly:
      return std::make_unique<EmitAssemblyAction>();

    case EmitObject:
      return std::make_unique<EmitObjAction>();

    default:
      break;
  }

  llvm_unreachable("Invalid program action");
  return nullptr;
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
    // Honor --help.
    if (ci->getFrontendOptions().showHelp) {
      options::getDriverOptTable().printHelp(
          llvm::outs(),
          "marco -mc1 [options] input-files", "MARCO Modelica frontend",
          options::MC1Option,
          llvm::opt::DriverFlag::HelpHidden,
          false);

      return true;
    }

    // Honor --version.
    if (ci->getFrontendOptions().showVersion) {
      options::printHelp();
      return true;
    }

    // If there were errors in processing arguments, don't do anything else.
    if (ci->getDiagnostics().hasErrors()) {
      return false;
    }

    // Create and execute the frontend action.
    std::unique_ptr<FrontendAction> act(createFrontendAction(*ci));

    if (!act) {
      return false;
    }

    return ci->executeAction(*act);
  }
}
