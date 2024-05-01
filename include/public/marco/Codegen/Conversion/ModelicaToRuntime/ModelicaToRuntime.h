#ifndef MARCO_CODEGEN_CONVERSION_MODELICATORUNTIME_MODELICATORUNTIME_H
#define MARCO_CODEGEN_CONVERSION_MODELICATORUNTIME_MODELICATORUNTIME_H

#include "mlir/Pass/Pass.h"

namespace mlir
{
#define GEN_PASS_DECL_MODELICATORUNTIMECONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createModelicaToRuntimeConversionPass();

  std::unique_ptr<mlir::Pass> createModelicaToRuntimeConversionPass(
      const ModelicaToRuntimeConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICATORUNTIME_MODELICATORUNTIME_H
