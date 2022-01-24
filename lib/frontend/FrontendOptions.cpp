#include "marco/frontend/FrontendOptions.h"

namespace marco::frontend
{
  bool mustBePreprocessed(llvm::StringRef suffix) {
    return suffix == "mo";
  }

  InputKind FrontendOptions::getInputKindForExtension(llvm::StringRef extension) {
    if (extension == "mo") {
      return Language::Modelica;
    }

    return Language::Unknown;
  }
}
