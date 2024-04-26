#ifndef MARCO_FRONTEND_INPUTFILE_H
#define MARCO_FRONTEND_INPUTFILE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstdint>

namespace marco::io
{
  enum class Language : uint8_t
  {
    Unknown,
    Modelica,
    BaseModelica,
    MLIR,
    LLVM_IR
  };

  enum class Format : uint8_t
  {
    Source,
    Object,
    Unknown
  };

  /// The kind of a file that we've been handed as an input.
  class InputKind
  {
    public:
      InputKind(
        Language language = Language::Unknown,
        Format format = Format::Unknown);

      Language getLanguage() const;

      Format getFormat() const;

      /// Get whether the input kind is fully-unknown.
      bool isUnknown() const;

      /// Return the appropriate input kind for a file.
      static InputKind getFromFullFileName(llvm::StringRef fileName);

      /// Return the appropriate input kind for a file extension.
      static InputKind getFromExtension(llvm::StringRef extension);

    private:
      Language language;
      Format format;
  };

  /// An input file to the frontend.
  class InputFile
  {
    public:
      InputFile();

      InputFile(llvm::StringRef file, InputKind kind);

      InputFile(const llvm::MemoryBuffer* buffer, InputKind kind);

      InputKind getKind() const;

      bool isEmpty() const;

      bool isFile() const;

      bool isBuffer() const;

      llvm::StringRef getFile() const;

      const llvm::MemoryBuffer* getBuffer() const;

    private:
      /// File name ("-" for standard input).
      std::string file;

      /// The input, if it comes from a buffer rather than a file. This object
      /// does not own the buffer, and the caller is responsible for ensuring
      /// that it outlives any users.
      const llvm::MemoryBuffer* buffer = nullptr;

      /// The kind of input, atm it contains language.
      InputKind kind;
  };
}

#endif // MARCO_FRONTEND_INPUTFILE_H
