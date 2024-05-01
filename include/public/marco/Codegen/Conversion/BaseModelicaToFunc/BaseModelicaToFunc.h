#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATOFUNC_BASEMODELICATOFUNC_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATOFUNC_BASEMODELICATOFUNC_H

#include "mlir/Pass/Pass.h"

namespace mlir
{
#define GEN_PASS_DECL_BASEMODELICATOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createBaseModelicaToFuncConversionPass();

  std::unique_ptr<mlir::Pass> createBaseModelicaToFuncConversionPass(
      const BaseModelicaToFuncConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATOFUNC_BASEMODELICATOFUNC_H
