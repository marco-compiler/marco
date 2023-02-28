#include "marco/Frontend/FrontendOptions.h"

namespace marco::frontend
{
  InputKind FrontendOptions::getInputKindForExtension(
      llvm::StringRef extension)
  {
    if (extension == "mo") {
      return Language::Modelica;
    }

    if (extension == "mlir") {
      return Language::MLIR;
    }

    if (extension == "bc" || extension == "ll") {
      return Language::LLVM_IR;
    }

    return Language::Unknown;
  }
}
