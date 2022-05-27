#include "marco/Diagnostic/Location.h"

using namespace ::marco;

namespace marco
{
  SourcePosition::SourcePosition(llvm::StringRef file, size_t line, size_t column)
      : file(std::make_shared<std::string>(file.str())),
        line(line),
        column(column)
  {
  }

  SourcePosition SourcePosition::unknown()
  {
    return SourcePosition("-", 0, 0);
  }

  SourceRange::SourceRange(SourcePosition begin, SourcePosition end)
    : begin(std::move(begin)),
      end(std::move(end))
  {
  }

  SourceRange SourceRange::unknown()
  {
    return SourceRange(SourcePosition::unknown(), SourcePosition::unknown());
  }
}
