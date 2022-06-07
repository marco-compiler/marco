#ifndef MARCO_DIAGNOSTIC_LOCATION_H
#define MARCO_DIAGNOSTIC_LOCATION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

namespace llvm
{
  class raw_ostream;
}

namespace marco
{
  class SourceFile
  {
    public:
      SourceFile(llvm::StringRef file, std::unique_ptr<llvm::MemoryBuffer> buffer);

      bool operator==(const SourceFile& other) const;

      llvm::StringRef filePath() const;
      const char* source() const;

    private:
      std::string filePath_;
      std::unique_ptr<llvm::MemoryBuffer> buffer_;
  };

  class SourcePosition
  {
    public:
      SourcePosition(std::shared_ptr<SourceFile> file, int64_t line, int64_t column);

      static SourcePosition unknown();

    public:
      std::shared_ptr<SourceFile> file;
      int64_t line;
      int64_t column;
  };

  class SourceRange
  {
    public:
      SourceRange(SourcePosition begin, SourcePosition end);

      static SourceRange unknown();

    public:
      SourcePosition begin;
      SourcePosition end;
  };
}

#endif // MARCO_DIAGNOSTIC_LOCATION_H
