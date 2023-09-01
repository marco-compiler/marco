#include "marco/IO/Command.h"
#include "llvm/Support/Program.h"

using namespace ::marco::io;

namespace marco::io
{
  Command::Command(llvm::StringRef programPath)
      : programPath(programPath.str())
  {
    args.push_back(programPath.str());
  }

  Command& Command::appendArg(const char* arg)
  {
    if (!llvm::StringRef(arg).empty()) {
      args.emplace_back(arg);
    }

    return *this;
  }

  Command& Command::appendArg(llvm::StringRef arg)
  {
    if (!arg.empty()) {
      args.push_back(arg.str());
    }

    return *this;
  }

  Command& Command::appendArgs(llvm::ArrayRef<const char*> newArgs)
  {
    for (const char* arg : newArgs) {
      if (!llvm::StringRef(arg).empty()) {
        args.emplace_back(arg);
      }
    }

    return *this;
  }

  Command& Command::setStdinRedirect(llvm::StringRef file)
  {
    stdinRedirect = file.str();
    return *this;
  }

  Command& Command::setStdoutRedirect(llvm::StringRef file)
  {
    stdoutRedirect = file.str();
    return *this;
  }

  Command& Command::setStderrRedirect(llvm::StringRef file)
  {
    stderrRedirect = file.str();
    return *this;
  }

  int Command::exec() const
  {
    assert(args.front() == programPath);

    llvm::SmallVector<llvm::StringRef> argsRef;
    llvm::SmallVector<llvm::Optional<llvm::StringRef>> redirects;

    for (const auto& arg : args) {
      argsRef.emplace_back(arg);
    }

    if (stdinRedirect) {
      redirects.push_back(llvm::StringRef(*stdinRedirect));
    } else {
      redirects.push_back(llvm::None);
    }

    if (stdoutRedirect) {
      redirects.push_back(llvm::StringRef(*stdoutRedirect));
    } else {
      redirects.push_back(llvm::None);
    }

    if (stderrRedirect) {
      redirects.push_back(llvm::StringRef(*stderrRedirect));
    } else {
      redirects.push_back(llvm::None);
    }

    /*
    llvm::outs() << "Running command\n";

    for (const auto& arg : argsRef) {
      llvm::outs() << arg << " ";
    }

    llvm::outs() << "\n";
    */

    return llvm::sys::ExecuteAndWait(
        programPath, argsRef, llvm::None, redirects);
  }
}
