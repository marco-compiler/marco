#include "marco/Frontend/FrontendAction.h"
#include "marco/Frontend/CompilerInstance.h"
#include "llvm/ADT/ScopeExit.h"

using namespace ::marco;
using namespace ::marco::frontend;
using namespace ::marco::io;

//===---------------------------------------------------------------------===//
// FrontendAction
//===---------------------------------------------------------------------===//

namespace marco::frontend {
FrontendAction::FrontendAction() : instance(nullptr) {}

FrontendAction::~FrontendAction() = default;

CompilerInstance &FrontendAction::getInstance() {
  assert(instance && "Compiler instance not registered");
  return *instance;
}

const CompilerInstance &FrontendAction::getInstance() const {
  assert(instance && "Compiler instance not registered");
  return *instance;
}

void FrontendAction::setInstance(CompilerInstance *value) { instance = value; }

llvm::ArrayRef<InputFile> FrontendAction::getCurrentInputs() const {
  return currentInputs;
}

llvm::SmallVector<std::string, 1> FrontendAction::getCurrentFiles() const {
  assert(!currentInputs.empty() && "No current files!");
  llvm::SmallVector<std::string, 1> result;

  for (const auto &currentInput : currentInputs) {
    assert(!currentInput.isEmpty() && "Empty current file");
    result.push_back(currentInput.getFile().str());
  }

  return result;
}

llvm::SmallVector<std::string, 1>
FrontendAction::getCurrentFilesOrBufferNames() const {
  assert(!currentInputs.empty() && "No current files!");
  llvm::SmallVector<std::string, 1> result;

  for (const auto &currentInput : currentInputs) {
    assert(!currentInput.isEmpty() && "Empty current file");

    if (currentInput.isFile()) {
      result.push_back(currentInput.getFile().str());
    } else {
      result.push_back(currentInput.getBuffer()->getBufferIdentifier().str());
    }
  }

  return result;
}

void FrontendAction::setCurrentInputs(llvm::ArrayRef<InputFile> inputs) {
  currentInputs.clear();

  for (const InputFile &inputFile : inputs) {
    currentInputs.push_back(inputFile);
  }
}

bool FrontendAction::prepareToExecute(CompilerInstance &ci) {
  return prepareToExecuteAction(ci);
}

bool FrontendAction::beginSourceFiles(CompilerInstance &ci,
                                      llvm::ArrayRef<InputFile> realInputs) {
  assert(!instance && "Already processing source files");

  setCurrentInputs(realInputs);
  setInstance(&ci);

  auto failureCleanup = llvm::make_scope_exit([&]() {
    ci.clearOutputFiles(true);
    setCurrentInputs({});
    setInstance(nullptr);
  });

  if (!beginInvocation()) {
    return false;
  }

  // Set up the file and source managers, if needed.
  if (!ci.hasFileManager()) {
    if (!ci.createFileManager()) {
      return false;
    }
  }

  if (!ci.hasSourceManager()) {
    ci.createSourceManager(ci.getFileManager());
  }

  // Inform the diagnostic client we are processing a source file.
  ci.getDiagnosticClient().BeginSourceFile(ci.getLanguageOptions());

  // Initialize the action.
  if (!beginSourceFilesAction()) {
    return false;
  }

  failureCleanup.release();
  return true;
}

llvm::Error FrontendAction::execute() {
  executeAction();
  return llvm::Error::success();
}

void FrontendAction::endSourceFiles() {
  CompilerInstance &ci = getInstance();

  // Inform the diagnostic client we are done with this source file.
  ci.getDiagnosticClient().EndSourceFile();

  // Finalize the action.
  endSourceFilesAction();

  // Cleanup the output streams, and erase the output files if instructed by
  // the FrontendAction.
  ci.clearOutputFiles(shouldEraseOutputFiles());

  setInstance(nullptr);
  setCurrentInputs({});
}

bool FrontendAction::shouldEraseOutputFiles() const {
  return getInstance().getDiagnostics().hasErrorOccurred();
}

bool FrontendAction::prepareToExecuteAction(CompilerInstance &ci) {
  return true;
}

bool FrontendAction::beginInvocation() { return true; }

bool FrontendAction::beginSourceFilesAction() { return true; }

void FrontendAction::endSourceFilesAction() {}
} // namespace marco::frontend
