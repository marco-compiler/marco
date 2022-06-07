#include "marco/Diagnostic/Location.h"

using namespace ::marco;

namespace marco
{
  SourceFile::SourceFile(llvm::StringRef filePath, std::unique_ptr<llvm::MemoryBuffer> buffer)
    : filePath_(filePath.str()), buffer_(std::move(buffer))
  {
  }

  bool SourceFile::operator==(const SourceFile& other) const
  {
    return filePath_ == other.filePath_;
  }

  llvm::StringRef SourceFile::filePath() const
  {
    return filePath_;
  }

  const char* SourceFile::source() const
  {
    assert(buffer_ != nullptr);
    return buffer_->getBufferStart();
  }

  SourcePosition::SourcePosition(std::shared_ptr<SourceFile> file, int64_t line, int64_t column)
      : file(file),
        line(line),
        column(column)
  {
  }

  SourcePosition SourcePosition::unknown()
  {
    auto file = std::make_unique<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(""));
    return SourcePosition(std::move(file), 0, 0);
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
