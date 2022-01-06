#include <clang/Driver/Options.h>
#include <llvm/Option/OptTable.h>
#include <llvm/Option/Option.h>
#include <llvm/Support/BuryPointer.h>
#include <llvm/Support/CommandLine.h>
#include <marco/frontend/CompilerInstance.h>
#include <marco/frontend/FrontendActions.h>
#include <marco/frontend/Options.h>
#include <marco/frontendTool/Utils.h>

namespace marco::frontend
{
  static std::unique_ptr<CompilerAction> CreateFrontendBaseAction(CompilerInstance& ci)
  {
    ActionKind action = ci.frontendOpts().programAction;

    switch (action) {
      case PrintAST:
        // TODO

      case EmitMLIR:
        // TODO

      case EmitLLVM:
        // TODO

      case EmitBitcode:
        // TODO

      default:
        break;
    }

    return 0;
  }

  std::unique_ptr<CompilerAction> CreateFrontendAction(CompilerInstance& ci)
  {
    // Create the underlying action.
    std::unique_ptr<CompilerAction> act = CreateFrontendBaseAction(ci);

    if (!act) {
      return nullptr;
    }

    return act;
  }

  bool ExecuteCompilerInvocation(CompilerInstance* instance)
  {
    // Honor -help
    if (instance->frontendOpts().showHelp) {
      marco::frontend::getDriverOptTable().printHelp(
          llvm::outs(),
          "marco-driver -mc1 [options] input-files", "MLIR Modelica compiler",
          false);

      return true;
    }

    // Honor -version
    if (instance->frontendOpts().showVersion) {
      llvm::cl::PrintVersionMessage();
      return true;
    }

    // Create and execute the frontend action
    std::unique_ptr<CompilerAction> act(CreateFrontendAction(*instance));

    if (!act) {
      return false;
    }

    return instance->ExecuteAction(*act);
  }
}
