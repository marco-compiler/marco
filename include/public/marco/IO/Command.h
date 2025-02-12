#ifndef MARCO_FRONTEND_COMMAND_H
#define MARCO_FRONTEND_COMMAND_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace marco::io {
class Command {
public:
  Command(llvm::StringRef programPath);

  Command &appendArg(const char *arg);
  Command &appendArg(llvm::StringRef arg);

  Command &appendArgs(llvm::ArrayRef<const char *> args);

  Command &setStdinRedirect(llvm::StringRef file);
  Command &setStdoutRedirect(llvm::StringRef file);
  Command &setStderrRedirect(llvm::StringRef file);

  int exec() const;

private:
  std::string programPath;
  llvm::SmallVector<std::string> args;
  std::optional<std::string> stdinRedirect = std::nullopt;
  std::optional<std::string> stdoutRedirect = std::nullopt;
  std::optional<std::string> stderrRedirect = std::nullopt;
};
} // namespace marco::io

#endif // MARCO_FRONTEND_COMMAND_H
