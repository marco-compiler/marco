#ifndef MARCO_FRONTEND_FRONTENDOPTIONS_H
#define MARCO_FRONTEND_FRONTENDOPTIONS_H

#include "marco/IO/InputFile.h"
#include "marco/VariableFilter/VariableFilter.h"
#include <string>

namespace marco::frontend
{
  enum ActionKind
  {
    InvalidAction = 0,

    InitOnly,

    // Emit a .bmo file.
    EmitBaseModelica,

    // Emit a .ast file.
    EmitAST,

    // Emit a .mlir file.
    EmitMLIR,

    // Emit a .ll file.
    EmitLLVMIR,

    // Emit a .bc file
    EmitLLVMBitcode,

    // Emit a .s file.
    EmitAssembly,

    // Emit a .o file.
    EmitObject
  };

  struct FrontendOptions
  {
    bool showHelp = false;

    bool showVersion = false;

    /// The input files and their types.
    std::vector<io::InputFile> inputs;

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

    bool multithreading = true;
  };
}

#endif // MARCO_FRONTEND_FRONTENDOPTIONS_H
