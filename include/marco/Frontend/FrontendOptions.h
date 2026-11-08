#ifndef MARCO_FRONTEND_FRONTENDOPTIONS_H
#define MARCO_FRONTEND_FRONTENDOPTIONS_H

#include "marco/IO/InputFile.h"
#include "marco/VariableFilter/VariableFilter.h"
#include <string>

namespace marco::frontend {
enum ActionKind {
  InvalidAction = 0,

  InitOnly,

  // Emit a .bmo file.
  EmitBaseModelica,

  // Emit a .ast file.
  EmitAST,

  // Emit a .mlir file.
  EmitMLIR,

  // Emit a .mo.mlir file, containing the flat model in the Modelica dialect.
  EmitMLIRModelica,

  // Emit a .bmo.mlir file, containing the model in the Base Modelica dialect.
  EmitMLIRBaseModelica,

  // Emit a .llvm.mlir file.
  EmitMLIRLLVM,

  // Emit a .ll file.
  EmitLLVMIR,

  // Emit a .bc file
  EmitLLVMBitcode,

  // Emit a .s file.
  EmitAssembly,

  // Emit a .o file.
  EmitObject
};

/// The frontend used to flatten the Modelica models.
enum class ModelicaFrontend {
  // OpenModelica, producing Base Modelica code.
  OMC,

  // MARCO's own frontend, based on the Modelica dialect.
  Native
};

struct FrontendOptions {
  bool showHelp = false;

  bool showVersion = false;

  /// The input files and their types.
  std::vector<io::InputFile> inputs;

  /// The output file, if any.
  std::string outputFile;

  /// The frontend action to perform.
  frontend::ActionKind programAction = InvalidAction;

  ModelicaFrontend modelicaFrontend = ModelicaFrontend::OMC;

  // OMC options
  bool omcBypass = false;
  std::string omcPath = "";
  std::vector<std::string> omcCustomArgs;

  std::string variableFilter;

  bool printModelInfo = false;

  // Whether to print statistics when the compilation finishes.
  bool printStatistics = false;

  bool multithreading = true;

  std::string printIRBeforePass = "";
  std::string printIRAfterPass = "";
  bool printIROnFailure = false;

  // Emit the IR needed for the verification of the simulation data.
  std::string verificationModelPath = "";

  bool shouldPrintIR() const;
};
} // namespace marco::frontend

#endif // MARCO_FRONTEND_FRONTENDOPTIONS_H
