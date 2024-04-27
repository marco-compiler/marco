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
      SourceFile(llvm::StringRef fileName);

      bool operator==(const SourceFile& other) const;

      llvm::StringRef getFileName() const;

      llvm::MemoryBuffer* getBuffer() const;

      void setMemoryBuffer(llvm::MemoryBuffer* buffer);

    private:
      std::string fileName;
      llvm::MemoryBuffer* buffer;
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
