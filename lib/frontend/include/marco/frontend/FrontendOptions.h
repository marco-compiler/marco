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

    PrintAST,
    EmitMLIR,
    EmitLLVM,
    EmitBitcode
  };

  /// \param suffix The file extension
  /// \return True if the file should be preprocessed
  bool mustBePreprocessed(llvm::StringRef suffix);

  enum class Language : uint8_t {
    Unknown,

    /// LLVM IR: we accept this so that we can run the optimizer on it,
    /// and compile it to assembly or object code.
    LLVM_IR,

    /// @{ Languages that the frontend can parse and compile.
    Modelica
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
      const llvm::MemoryBuffer *buffer_ = nullptr;

      /// The kind of input, atm it contains language
      InputKind kind_;

      /// Must this file be preprocessed? Note that in Flang the preprocessor is
      /// always run. This flag is used to control whether predefined and command
      /// line preprocessor macros are enabled or not. In practice, this is
      /// sufficient to implement gfortran`s logic controlled with `-cpp/-nocpp`.
      unsigned mustBePreprocessed_ : 1;

    public:
      FrontendInputFile() = default;
      FrontendInputFile(llvm::StringRef file, InputKind kind)
          : file_(file.str()), kind_(kind) {

        // Based on the extension, decide whether this is a fixed or free form
        // file.
        auto pathDotIndex{file.rfind(".")};
        std::string pathSuffix{file.substr(pathDotIndex + 1)};
        mustBePreprocessed_ = mustBePreprocessed(pathSuffix);
      }

      FrontendInputFile(const llvm::MemoryBuffer *buffer, InputKind kind)
          : buffer_(buffer), kind_(kind) {}

      InputKind kind() const { return kind_; }

      bool IsEmpty() const { return file_.empty() && buffer_ == nullptr; }
      bool IsFile() const { return !IsBuffer(); }
      bool IsBuffer() const { return buffer_ != nullptr; }
      bool MustBePreprocessed() const { return mustBePreprocessed_; }

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
