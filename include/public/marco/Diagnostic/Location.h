#ifndef MARCO_DIAGNOSTIC_LOCATION_H
#define MARCO_DIAGNOSTIC_LOCATION_H

#include "llvm/ADT/StringRef.h"
#include <memory>

namespace llvm
{
  class raw_ostream;
}

namespace marco
{
  class SourcePosition
  {
    public:
      SourcePosition(llvm::StringRef file, size_t line, size_t column);

      static SourcePosition unknown();

    public:
      std::shared_ptr<std::string> file;
      size_t line;
      size_t column;
  };

  /*
  llvm::raw_ostream& operator<<(
      llvm::raw_ostream& stream, const SourcePosition& obj);

  std::string toString(const SourcePosition& obj);
   */

  class SourceRange
  {
    public:
      SourceRange(SourcePosition begin, SourcePosition end);

      static SourceRange unknown();

      //SourcePosition getStartPosition() const;
      //void extendEnd(SourceRange to);

      //void printLines(llvm::raw_ostream& os) const;

    public:
      SourcePosition begin;
      SourcePosition end;
  };
}

#endif // MARCO_DIAGNOSTIC_LOCATION_H
