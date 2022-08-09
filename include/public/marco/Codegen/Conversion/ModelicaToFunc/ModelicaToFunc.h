#ifndef MARCO_CODEGEN_CONVERSION_MODELICATOFUNC_MODELICATOFUNC_H
#define MARCO_CODEGEN_CONVERSION_MODELICATOFUNC_MODELICATOFUNC_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
  struct ModelicaToFuncOptions
  {
    unsigned int bitWidth = 64;

    static const ModelicaToFuncOptions& getDefaultOptions();
  };

  std::unique_ptr<mlir::Pass> createModelicaToFuncPass(
      ModelicaToFuncOptions options = ModelicaToFuncOptions::getDefaultOptions());
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICATOFUNC_MODELICATOFUNC_H
