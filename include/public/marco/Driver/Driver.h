#ifndef MARCO_FRONTEND_DRIVER_H
#define MARCO_FRONTEND_DRIVER_H

#include "marco/IO/InputFile.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/ArgList.h"

namespace marco::driver
{
  class Driver
  {
    public:
      Driver(llvm::StringRef executablePath);

      int run(llvm::ArrayRef<const char*> argv) const;

    private:
      int executeMC1Tool(
        const llvm::opt::ArgList& args,
        llvm::ArrayRef<io::InputFile> inputFiles,
        llvm::StringRef outputFileName) const;

      void collectLibraryPaths(
          llvm::StringRef pathsStr,
          llvm::SmallVectorImpl<std::string>& paths) const;

    private:
      std::string executablePath;
  };
}

#endif // MARCO_FRONTEND_DRIVER_H
