#include "marco/Lexer/Location.h"

using namespace ::marco;

namespace marco {
SourceFile::SourceFile(llvm::StringRef fileName) : fileName(fileName.str()) {}

bool SourceFile::operator==(const SourceFile &other) const {
  return fileName == other.fileName;
}

llvm::StringRef SourceFile::getFileName() const { return fileName; }

llvm::MemoryBuffer *SourceFile::getBuffer() const {
  assert(buffer && "The source file has no buffer");
  return buffer;
}

void SourceFile::setMemoryBuffer(llvm::MemoryBuffer *buffer) {
  this->buffer = buffer;
}

SourcePosition::SourcePosition(std::shared_ptr<SourceFile> file, int64_t line,
                               int64_t column)
    : file(std::move(file)), line(line), column(column) {}

SourcePosition SourcePosition::unknown() {
  auto file = std::make_unique<SourceFile>("-");
  return SourcePosition(std::move(file), 0, 0);
}

SourceRange::SourceRange(SourcePosition begin, SourcePosition end)
    : begin(std::move(begin)), end(std::move(end)) {}

SourceRange SourceRange::unknown() {
  return SourceRange(SourcePosition::unknown(), SourcePosition::unknown());
}
} // namespace marco
