#include "clang/Driver/Options.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/InitLLVM.h"
#include "clang/Driver/Driver.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Host.h"
#include "marco/Frontend/TextDiagnosticPrinter.h"

#define BUG_REPORT_URL "https://github.com/modelica-polimi/marco/issues/"

static const char *BugReportMsg =
    "PLEASE submit a bug report to " BUG_REPORT_URL
    " and include the crash backtrace.\n";

extern int mc1_main(llvm::ArrayRef<const char *> argv, const char *argv0);

std::string getExecutablePath(const char *argv0) {
  // This just needs to be some symbol in the binary
  void *p = (void *)(intptr_t)getExecutablePath;
  return llvm::sys::fs::getMainExecutable(argv0, p);
}

// This lets us create the DiagnosticsEngine with a properly-filled-out
// DiagnosticOptions instance
static clang::DiagnosticOptions* createAndPopulateDiagOpts(llvm::ArrayRef<const char *> argv) {
  auto *diagOpts = new clang::DiagnosticOptions;

  // Ignore missingArgCount and the return value of ParseDiagnosticArgs.
  // Any errors that would be diagnosed here will also be diagnosed later,
  // when the DiagnosticsEngine actually exists.
  unsigned missingArgIndex, missingArgCount;
  llvm::opt::InputArgList args = clang::driver::getDriverOptTable().ParseArgs(
      argv.slice(1), missingArgIndex, missingArgCount,
      llvm::opt::Visibility(clang::driver::options::MarcoOption));

  //This is used by flang, but we don't use it
  //(void)marco::frontend::parseDiagnosticArgs(*diagOpts, args);

  return diagOpts;
}

static void ExpandResponseFiles(llvm::StringSaver &saver,
                                llvm::SmallVectorImpl<const char *> &args) {
  // We're defaulting to the GNU syntax, since we don't have a CL mode.
  llvm::cl::TokenizerCallback tokenizer = &llvm::cl::TokenizeGNUCommandLine;
  llvm::cl::ExpansionContext ExpCtx(saver.getAllocator(), tokenizer);
  if (llvm::Error Err = ExpCtx.expandResponseFiles(args)) {
    llvm::errs() << toString(std::move(Err)) << '\n';
  }
}

static int executeMC1Tool(llvm::SmallVectorImpl<const char *> &argv) {
  llvm::StringRef tool = argv[1];
  if (tool == "-mc1")
    return mc1_main(llvm::ArrayRef(argv).slice(2), argv[0]);

  // Reject unknown tools.
  llvm::errs() << "error: unknown integrated tool '" << tool << "'. "
               << "Valid tools include '-mc1'.\n";
  return 1;
}

int main(int argc, const char** argv)
{
  // Initialize variables to call the driver
  llvm::InitLLVM x(argc, argv);
  llvm::SmallVector<const char *, 256> args(argv, argv + argc);

  clang::driver::ParsedClangName targetAndMode("marco", "--driver-mode=marco");
  std::string driverPath = getExecutablePath(args[0]);

  llvm::BumpPtrAllocator a;
  llvm::StringSaver saver(a);
  ExpandResponseFiles(saver, args);

  llvm::setBugReportMsg(BugReportMsg);

  // Check if marco is in the frontend mode
  auto firstArg = std::find_if(args.begin() + 1, args.end(),
                               [](const char *a) { return a != nullptr; });
  if (firstArg != args.end()) {
    // Call mc1 frontend
    if (llvm::StringRef(args[1]).equals("-mc1")) {
      return executeMC1Tool(args);
    }
  }

  // Not in the frontend mode - continue in the compiler driver mode.

  // Create DiagnosticsEngine for the compiler driver
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts =
      createAndPopulateDiagOpts(args);
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(
      new clang::DiagnosticIDs());
  marco::frontend::TextDiagnosticPrinter *diagClient =
      new marco::frontend::TextDiagnosticPrinter(llvm::errs(), &*diagOpts);

  diagClient->setPrefix(
      std::string(llvm::sys::path::stem(getExecutablePath(args[0]))));

  clang::DiagnosticsEngine diags(diagID, &*diagOpts, diagClient);

  // Prepare the driver
  clang::driver::Driver theDriver(driverPath,
                                  llvm::sys::getDefaultTargetTriple(), diags,
                                  "marco LLVM compiler");
  theDriver.setTargetAndMode(targetAndMode);
  std::unique_ptr<clang::driver::Compilation> c(
      theDriver.BuildCompilation(args));
  llvm::SmallVector<std::pair<int, const clang::driver::Command *>, 4>
      failingCommands;

  // Run the driver
  int res = 1;
  bool isCrash = false;
  res = theDriver.ExecuteCompilation(*c, failingCommands);

  for (const auto &p : failingCommands) {
    int commandRes = p.first;
    const clang::driver::Command *failingCommand = p.second;
    if (!res)
      res = commandRes;

    // If result status is < 0 (e.g. when sys::ExecuteAndWait returns -1),
    // then the driver command signalled an error. On Windows, abort will
    // return an exit code of 3. In these cases, generate additional diagnostic
    // information if possible.
    isCrash = commandRes < 0;
#ifdef _WIN32
    isCrash |= commandRes == 3;
#endif
    if (isCrash) {
      theDriver.generateCompilationDiagnostics(*c, *failingCommand);
      break;
    }
  }

  diags.getClient()->finish();

  // If we have multiple failing commands, we return the result of the first
  // failing command.
  return res;
}
