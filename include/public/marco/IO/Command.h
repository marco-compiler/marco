#ifndef MARCO_FRONTEND_COMMAND_H
#define MARCO_FRONTEND_COMMAND_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"

namespace marco::io
{
  class Command
  {
    public:
      Command(llvm::StringRef programPath);

      Command& appendArg(const char* arg);
      Command& appendArg(llvm::StringRef arg);

      Command& appendArgs(llvm::ArrayRef<const char*> args);

      Command& setStdinRedirect(llvm::StringRef file);
      Command& setStdoutRedirect(llvm::StringRef file);
      Command& setStderrRedirect(llvm::StringRef file);

      int exec() const;

    private:
      std::string programPath;
      llvm::SmallVector<std::string> args;
      llvm::Optional<std::string> stdinRedirect = llvm::None;
      llvm::Optional<std::string> stdoutRedirect = llvm::None;
      llvm::Optional<std::string> stderrRedirect = llvm::None;
  };
}

#endif //MARCO_FRONTEND_COMMAND_H
