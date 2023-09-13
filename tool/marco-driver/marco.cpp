#include "marco/Driver/Driver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace ::marco;

extern int mc1_main(llvm::ArrayRef<const char*> argv, const char* argv0);

static int executeMC1Tool(llvm::SmallVectorImpl<const char*>& argv)
{
  llvm::StringRef tool = argv[1];

  if (tool == "-mc1") {
    return mc1_main(llvm::ArrayRef(argv).slice(2), argv[0]);
  }

  // Reject unknown tools.
  llvm::errs() << "error: unknown integrated tool '" << tool << "'. "
               << "Valid tools include '-mc1'.\n";

  return 1;
}

static std::string getExecutablePath(const char* argv0)
{
  // This just needs to be some symbol in the binary.
  void *p = (void*)(intptr_t) getExecutablePath;
  return llvm::sys::fs::getMainExecutable(argv0, p);
}

int main(int argc, const char** argv)
{
  llvm::InitLLVM x(argc, argv);
  llvm::SmallVector<const char*, 256> args(argv, argv + argc);
  std::string driverPath = getExecutablePath(args[0]);

  // Check if MARCO is in the frontend mode.
  auto firstArg =
      std::find_if(args.begin() + 1, args.end(), [](const char *a) {
        return a != nullptr;
      });

  if (firstArg != args.end()) {
    // Call the frontend.
    if (llvm::StringRef(args[1]).startswith("-mc1")) {
      return executeMC1Tool(args);
    }
  }

  driver::Driver driver(driverPath);
  return driver.run(llvm::ArrayRef(args).slice(1));
}
