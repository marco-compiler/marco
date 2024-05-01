#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATORUNTIME_BASEMODELICATORUNTIME_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATORUNTIME_BASEMODELICATORUNTIME_H

#include "mlir/Pass/Pass.h"

namespace mlir
{
#define GEN_PASS_DECL_BASEMODELICATORUNTIMECONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createBaseModelicaToRuntimeConversionPass();

  std::unique_ptr<mlir::Pass> createBaseModelicaToRuntimeConversionPass(
      const BaseModelicaToRuntimeConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATORUNTIME_BASEMODELICATORUNTIME_H
