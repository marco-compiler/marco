#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATOARITH_BASEMODELICATOARITH_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATOARITH_BASEMODELICATOARITH_H

#include "mlir/Pass/Pass.h"
#include "llvm/IR/DataLayout.h"

namespace mlir
{
#define GEN_PASS_DECL_BASEMODELICATOARITHCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createBaseModelicaToArithConversionPass();

  std::unique_ptr<mlir::Pass> createBaseModelicaToArithConversionPass(
      const BaseModelicaToArithConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATOARITH_BASEMODELICATOARITH_H
