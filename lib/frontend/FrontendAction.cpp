#include <marco/frontend/CompilerInstance.h>
#include <marco/frontend/FrontendAction.h>
#include <marco/frontend/FrontendActions.h>
#include <marco/frontend/FrontendOptions.h>
#include <clang/Basic/DiagnosticFrontend.h>
#include <llvm/Support/Errc.h>
#include <llvm/Support/VirtualFileSystem.h>

namespace marco::frontend
{
  void FrontendAction::set_currentInput(const FrontendInputFile &currentInput) {
    this->currentInput_ = currentInput;
  }

  // Call this method if BeginSourceFile fails.
  // Deallocate compiler instance, input and output descriptors
  static void BeginSourceFileCleanUp(FrontendAction &fa, CompilerInstance &ci) {
    ci.ClearOutputFiles(/*EraseFiles=*/true);
    fa.set_currentInput(FrontendInputFile());
    fa.set_instance(nullptr);
  }

  bool FrontendAction::BeginSourceFile(
      CompilerInstance &ci, const FrontendInputFile &realInput) {

    FrontendInputFile input(realInput);

    // Return immediately if the input file does not exist or is not a file. Note
    // that we cannot check this for input from stdin.
    if (input.file() != "-") {
      if (!llvm::sys::fs::is_regular_file(input.file())) {
        // Create an diagnostic ID to report
        unsigned diagID;
        if (llvm::vfs::getRealFileSystem()->exists(input.file())) {
          ci.diagnostics().Report(clang::diag::err_fe_error_reading)
              << input.file();
          diagID = ci.diagnostics().getCustomDiagID(
              clang::DiagnosticsEngine::Error, "%0 is not a regular file");
        } else {
          diagID = ci.diagnostics().getCustomDiagID(
              clang::DiagnosticsEngine::Error, "%0 does not exist");
        }

        // Report the diagnostic and return
        ci.diagnostics().Report(diagID) << input.file();
        BeginSourceFileCleanUp(*this, ci);
        return false;
      }
    }

    assert(!instance_ && "Already processing a source file!");
    assert(!realInput.IsEmpty() && "Unexpected empty filename!");
    set_currentInput(realInput);
    set_instance(&ci);

    /*
    if (!ci.HasAllSources()) {
      BeginSourceFileCleanUp(*this, ci);
      return false;
    }

    if (!BeginSourceFileAction()) {
      BeginSourceFileCleanUp(*this, ci);
      return false;
    }
     */

    return true;
  }

  bool FrontendAction::ShouldEraseOutputFiles() {
    return instance().diagnostics().hasErrorOccurred();
  }

  llvm::Error FrontendAction::Execute() {
    ExecuteAction();

    return llvm::Error::success();
  }

  void FrontendAction::EndSourceFile() {
    CompilerInstance &ci = instance();

    // Cleanup the output streams, and erase the output files if instructed by the
    // FrontendAction.
    ci.ClearOutputFiles(/*EraseFiles=*/ShouldEraseOutputFiles());

    set_instance(nullptr);
    set_currentInput(FrontendInputFile());
  }

  bool FrontendAction::RunParse() {
    /*
    CompilerInstance &ci = this->instance();

    // Parse. In case of failure, report and return.
    ci.parsing().Parse(llvm::outs());

    if (reportFatalParsingErrors()) {
      return false;
    }

    // Report the diagnostics from parsing
    ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());
     */

    return true;
  }

  template <unsigned N>
  bool FrontendAction::reportFatalErrors(const char (&message)[N]) {
    /*
    if (!instance_->parsing().messages().empty() &&
        (instance_->invocation().warnAsErr() ||
            instance_->parsing().messages().AnyFatalError())) {
      const unsigned diagID = instance_->diagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, message);
      instance_->diagnostics().Report(diagID) << GetCurrentFileOrBufferName();
      instance_->parsing().messages().Emit(
          llvm::errs(), instance_->allCookedSources());
      return true;
    }
    return false;
     */
    return false;
  }
}
