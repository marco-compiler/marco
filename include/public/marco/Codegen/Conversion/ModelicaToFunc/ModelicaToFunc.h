#ifndef MARCO_CODEGEN_CONVERSION_MODELICATOFUNC_MODELICATOFUNC_H
#define MARCO_CODEGEN_CONVERSION_MODELICATOFUNC_MODELICATOFUNC_H

#include "mlir/Pass/Pass.h"

namespace mlir
{
#define GEN_PASS_DECL_MODELICATOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createModelicaToFuncConversionPass();

  std::unique_ptr<mlir::Pass> createModelicaToFuncConversionPass(const ModelicaToFuncConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICATOFUNC_MODELICATOFUNC_H
