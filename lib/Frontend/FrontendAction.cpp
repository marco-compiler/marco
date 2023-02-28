#include "marco/Frontend/FrontendAction.h"
#include "marco/Frontend/CompilerInstance.h"
#include "llvm/ADT/ScopeExit.h"

using namespace ::marco;
using namespace ::marco::diagnostic;
using namespace ::marco::frontend;

//===---------------------------------------------------------------------===//
// FrontendAction
//===---------------------------------------------------------------------===//

namespace marco::frontend
{
  FrontendAction::FrontendAction()
      : instance(nullptr)
  {
  }

  FrontendAction::~FrontendAction() = default;

  CompilerInstance& FrontendAction::getInstance()
  {
    assert(instance && "Compiler instance not registered");
    return *instance;
  }

  const CompilerInstance& FrontendAction::getInstance() const
  {
    assert(instance && "Compiler instance not registered");
    return *instance;
  }

  void FrontendAction::setInstance(CompilerInstance* value)
  {
    instance = value;
  }

  const FrontendInputFile& FrontendAction::getCurrentInput() const
  {
    return currentInput;
  }

  llvm::StringRef FrontendAction::getCurrentFile() const
  {
    assert(!currentInput.isEmpty() && "No current file!");
    return currentInput.getFile();
  }

  llvm::StringRef FrontendAction::getCurrentFileOrBufferName() const
  {
    assert(!currentInput.isEmpty() && "No current file!");

    return currentInput.isFile()
        ? currentInput.getFile()
        : currentInput.getBuffer()->getBufferIdentifier();
  }

  void FrontendAction::setCurrentInput(const FrontendInputFile& input)
  {
    this->currentInput = input;
  }

  bool FrontendAction::prepareToExecute(CompilerInstance& ci)
  {
    return prepareToExecuteAction(ci);
  }

  bool FrontendAction::beginSourceFile(
      CompilerInstance& ci, const FrontendInputFile& realInput)
  {
    FrontendInputFile input(realInput);
    assert(!instance && "Already processing a source file");
    assert(!input.isEmpty() && "Unexpected empty file name");

    setCurrentInput(input);
    setInstance(&ci);

    auto failureCleanup = llvm::make_scope_exit([&]() {
      ci.clearOutputFiles(true);
      setCurrentInput(FrontendInputFile());
      setInstance(nullptr);
    });

    if (!beginInvocation()) {
      return false;
    }

    // Initialize the action.
    if (!beginSourceFileAction()) {
      return false;
    }

    failureCleanup.release();
    return true;
  }

  llvm::Error FrontendAction::execute()
  {
    executeAction();
    return llvm::Error::success();
  }

  void FrontendAction::endSourceFile()
  {
    CompilerInstance& ci = getInstance();

    // Finalize the action.
    endSourceFileAction();

    // Cleanup the output streams, and erase the output files if instructed by
    // the FrontendAction.
    ci.clearOutputFiles(shouldEraseOutputFiles());

    setInstance(nullptr);
    setCurrentInput(FrontendInputFile());
  }

  bool FrontendAction::shouldEraseOutputFiles() const
  {
    return getInstance().getDiagnostics().hasErrors();
  }

  bool FrontendAction::prepareToExecuteAction(CompilerInstance& ci)
  {
    return true;
  }

  bool FrontendAction::beginInvocation()
  {
    return true;
  }

  bool FrontendAction::beginSourceFileAction()
  {
    return true;
  }

  void FrontendAction::endSourceFileAction()
  {
  }
}
