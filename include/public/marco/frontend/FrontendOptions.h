#ifndef MARCO_FRONTEND_FRONTENDOPTIONS_H
#define MARCO_FRONTEND_FRONTENDOPTIONS_H

//#include "flang/Common/Fortran-features.h"
//#include "flang/Parser/characters.h"
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>

#include <cstdint>
#include <string>

namespace marco::frontend
{
  enum ActionKind {
    InvalidAction = 0,

    InitOnly,
    EmitAST,
    EmitModelicaDialect,
    EmitLLVMDialect,
    EmitLLVMIR,
    EmitBitcode
  };

  /// \param suffix The file extension
  /// \return True if the file should be preprocessed
  bool mustBePreprocessed(llvm::StringRef suffix);

  enum class Language : uint8_t {
    Unknown,
    Modelica,
    MLIR
  };

  /// The kind of a file that we've been handed as an input.
  class InputKind {
    private:
      Language lang_;

    public:
      /// The input file format.
      enum Format { Source, ModuleMap, Precompiled };

      constexpr InputKind(Language l = Language::Unknown) : lang_(l) {}

      Language GetLanguage() const { return static_cast<Language>(lang_); }

      /// Is the input kind fully-unknown?
      bool IsUnknown() const { return lang_ == Language::Unknown; }
  };

  /**
   * An input file to the frontend.
   */
  class FrontendInputFile {
      // File name ("-" for standard input)
      std::string file_;

      /// The input, if it comes from a buffer rather than a file. This object
      /// does not own the buffer, and the caller is responsible for ensuring
      /// that it outlives any users.
      const llvm::MemoryBuffer* buffer_ = nullptr;

      /// The kind of input, atm it contains language
      InputKind kind_;

    public:
      FrontendInputFile() = default;
      FrontendInputFile(llvm::StringRef file, InputKind kind)
          : file_(file.str()), kind_(kind) {
      }

      FrontendInputFile(const llvm::MemoryBuffer *buffer, InputKind kind)
          : buffer_(buffer), kind_(kind) {}

      InputKind kind() const { return kind_; }

      bool IsEmpty() const { return file_.empty() && buffer_ == nullptr; }
      bool IsFile() const { return !IsBuffer(); }
      bool IsBuffer() const { return buffer_ != nullptr; }

      llvm::StringRef file() const {
        assert(IsFile());
        return file_;
      }

      const llvm::MemoryBuffer *buffer() const {
        assert(IsBuffer() && "Requested buffer_, but it is empty!");
        return buffer_;
      }
  };

  /// FrontendOptions - Options for controlling the behavior of the frontend.
  struct FrontendOptions {
    FrontendOptions() {}

    bool showHelp = false;

    bool showVersion = false;

    /// The input files and their types.
    std::vector<FrontendInputFile> inputs;

    /// The output file, if any.
    std::string outputFile;

    /// The frontend action to perform.
    frontend::ActionKind programAction;

    // Return the appropriate input kind for a file extension. For example,
    /// "*.f" would return Language::Fortran.
    ///
    /// \return The input kind for the extension, or Language::Unknown if the
    /// extension is not recognized.
    static InputKind GetInputKindForExtension(llvm::StringRef extension);

    //virtual llvm::StringRef outputFile() const;
  };
}

#endif // MARCO_FRONTEND_FRONTENDOPTIONS_H
