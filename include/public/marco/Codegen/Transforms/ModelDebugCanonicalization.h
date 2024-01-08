#ifndef MARCO_CODEGEN_TRANSFORMS_MODELDEBUGCANONICALIZATIONPASS_H
#define MARCO_CODEGEN_TRANSFORMS_MODELDEBUGCANONICALIZATIONPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_MODELDEBUGCANONICALIZATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createModelDebugCanonicalizationPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELDEBUGCANONICALIZATIONPASS_H
