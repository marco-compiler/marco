#include "marco/IO/InputFile.h"

using namespace ::marco::io;

namespace marco::io {
InputKind::InputKind(Language language, Format format)
    : language(language), format(format) {}

Language InputKind::getLanguage() const {
  return static_cast<Language>(language);
}

Format InputKind::getFormat() const { return static_cast<Format>(format); }

bool InputKind::isUnknown() const {
  return language == Language::Unknown && format == Format::Unknown;
}

InputKind InputKind::getFromFullFileName(llvm::StringRef fileName) {
  return getFromExtension(llvm::StringRef(fileName).rsplit('.').second);
}

InputKind InputKind::getFromExtension(llvm::StringRef extension) {
  if (extension == "mo") {
    return {Language::Modelica, Format::Source};
  }

  if (extension == "mlir") {
    return {Language::MLIR, Format::Source};
  }

  if (extension == "bc" || extension == "ll") {
    return {Language::LLVM_IR, Format::Source};
  }

  if (extension == ".o") {
    return {Language::Unknown, Format::Object};
  }

  return {Language::Unknown, Format::Unknown};
}

InputFile::InputFile() = default;

InputFile::InputFile(llvm::StringRef file, InputKind kind)
    : file(file.str()), kind(kind) {}

InputFile::InputFile(const llvm::MemoryBuffer *buffer, InputKind kind)
    : buffer(buffer), kind(kind) {}

InputKind InputFile::getKind() const { return kind; }

bool InputFile::isEmpty() const { return file.empty() && buffer == nullptr; }

bool InputFile::isFile() const { return !isBuffer(); }

bool InputFile::isBuffer() const { return buffer != nullptr; }

llvm::StringRef InputFile::getFile() const {
  assert(isFile());
  return file;
}

const llvm::MemoryBuffer *InputFile::getBuffer() const {
  assert(isBuffer() && "Requested buffer, but it is empty!");
  return buffer;
}
} // namespace marco::io
