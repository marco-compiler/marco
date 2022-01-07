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
  static std::unique_ptr<FrontendAction> CreateFrontendBaseAction(CompilerInstance& ci)
  {
    ActionKind action = ci.frontendOpts().programAction;

    switch (action) {
      case InitOnly:
        return std::make_unique<InitOnlyAction>();

      case EmitAST:
        return std::make_unique<EmitASTAction>();

      case EmitModelicaDialect:
        return std::make_unique<EmitModelicaDialectAction>();

      case EmitLLVMDialect:
        return std::make_unique<EmitLLVMDialectAction>();

      case EmitLLVMIR:
        return std::make_unique<EmitLLVMIRAction>();

      case EmitBitcode:
        return std::make_unique<EmitBitcodeAction>();

      default:
        break;
    }

    return 0;
  }

  std::unique_ptr<FrontendAction> CreateFrontendAction(CompilerInstance& ci)
  {
    // Create the underlying action.
    std::unique_ptr<FrontendAction> act = CreateFrontendBaseAction(ci);

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
    std::unique_ptr<FrontendAction> act(CreateFrontendAction(*instance));

    if (!act) {
      return false;
    }

    return instance->ExecuteAction(*act);
  }
}
