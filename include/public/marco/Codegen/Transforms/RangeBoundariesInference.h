#ifndef MARCO_CODEGEN_TRANSFORMS_RANGEBOUNDARIESINFERENCE_H
#define MARCO_CODEGEN_TRANSFORMS_RANGEBOUNDARIESINFERENCE_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_RANGEBOUNDARIESINFERENCEPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createRangeBoundariesInferencePass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_RANGEBOUNDARIESINFERENCE_H
