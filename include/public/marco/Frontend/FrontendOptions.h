#ifndef MARCO_FRONTEND_FRONTENDOPTIONS_H
#define MARCO_FRONTEND_FRONTENDOPTIONS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "marco/VariableFilter/VariableFilter.h"
#include <cstdint>
#include <string>

namespace marco::frontend
{
  enum ActionKind
  {
    InvalidAction = 0,

    InitOnly,
    EmitFlattened,
    EmitAST,
    EmitFinalAST,

    // Emit a .mlir file.
    EmitMLIR,

    // Emit a .ll file.
    EmitLLVMIR,

    /// Emit a .bc file
    EmitLLVMBitcode,

    // Emit a .s file.
    EmitAssembly,

    // Emit a .o file.
    EmitObject
  };

  enum class Language : uint8_t
  {
    Unknown,
    Modelica,
    MLIR,
    LLVM_IR
  };

  /// The kind of a file that we've been handed as an input.
  class InputKind
  {
    private:
      Language lang;

    public:
      /// The input file format.
      enum Format { Source, ModuleMap, Precompiled };

      constexpr InputKind(Language l = Language::Unknown) : lang(l) {}

      Language getLanguage() const { return static_cast<Language>(lang); }

      /// Is the input kind fully-unknown?
      bool isUnknown() const { return lang == Language::Unknown; }
  };

  /// An input file to the frontend.
  class FrontendInputFile
  {
      /// File name ("-" for standard input).
      std::string file;

      /// The input, if it comes from a buffer rather than a file. This object
      /// does not own the buffer, and the caller is responsible for ensuring
      /// that it outlives any users.
      const llvm::MemoryBuffer* buffer = nullptr;

      /// The kind of input, atm it contains language.
      InputKind kind;

    public:
      FrontendInputFile() = default;

      FrontendInputFile(llvm::StringRef file, InputKind kind)
          : file(file.str()), kind(kind)
      {
      }

      FrontendInputFile(const llvm::MemoryBuffer* buffer, InputKind kind)
          : buffer(buffer), kind(kind)
      {
      }

      InputKind getKind() const { return kind; }

      bool isEmpty() const { return file.empty() && buffer == nullptr; }

      bool isFile() const { return !isBuffer(); }

      bool isBuffer() const { return buffer != nullptr; }

      llvm::StringRef getFile() const
      {
        assert(isFile());
        return file;
      }

      const llvm::MemoryBuffer* getBuffer() const
      {
        assert(isBuffer() && "Requested buffer, but it is empty!");
        return buffer;
      }
  };

  struct FrontendOptions
  {
    bool showHelp = false;

    bool showVersion = false;

    /// The input files and their types.
    std::vector<FrontendInputFile> inputs;

    /// The output file, if any.
    std::string outputFile;

    /// The frontend action to perform.
    frontend::ActionKind programAction = InvalidAction;

    // OMC options
    bool omcBypass = false;
    std::string omcPath = "";
    std::vector<std::string> omcCustomArgs;

    std::string variablesFilter;

    // Whether to print statistics when the compilation finishes.
    bool printStatistics = false;

    /// Return the appropriate input kind for a file extension.
    static InputKind getInputKindForExtension(llvm::StringRef extension);
  };
}

#endif // MARCO_FRONTEND_FRONTENDOPTIONS_H
