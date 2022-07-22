#ifndef MARCO_CODEGEN_CONVERSION_MODELICATOARITH_MODELICATOARITH_H
#define MARCO_CODEGEN_CONVERSION_MODELICATOARITH_MODELICATOARITH_H

#include "mlir/Pass/Pass.h"
#include "llvm/IR/DataLayout.h"

namespace marco::codegen
{
  struct ModelicaToArithOptions
  {
    unsigned int bitWidth = 64;
    bool assertions = true;

    llvm::DataLayout dataLayout = llvm::DataLayout("");

    static const ModelicaToArithOptions& getDefaultOptions();
  };

  std::unique_ptr<mlir::Pass> createModelicaToArithPass(
      ModelicaToArithOptions options = ModelicaToArithOptions::getDefaultOptions());
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICATOARITH_MODELICATOARITH_H
