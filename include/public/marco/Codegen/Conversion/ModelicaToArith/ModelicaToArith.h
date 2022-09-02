#ifndef MARCO_CODEGEN_CONVERSION_MODELICATOARITH_MODELICATOARITH_H
#define MARCO_CODEGEN_CONVERSION_MODELICATOARITH_MODELICATOARITH_H

#include "mlir/Pass/Pass.h"
#include "llvm/IR/DataLayout.h"

namespace mlir
{
#define GEN_PASS_DECL_MODELICATOARITHCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createModelicaToArithConversionPass();

  std::unique_ptr<mlir::Pass> createModelicaToArithConversionPass(const ModelicaToArithConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICATOARITH_MODELICATOARITH_H
