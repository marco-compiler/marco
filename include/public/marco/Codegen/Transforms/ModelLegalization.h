#ifndef MARCO_CODEGEN_TRANSFORMS_MODELLEGALIZATION_H
#define MARCO_CODEGEN_TRANSFORMS_MODELLEGALIZATION_H

#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_MODELLEGALIZATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createModelLegalizationPass();

  std::unique_ptr<mlir::Pass> createModelLegalizationPass(const ModelLegalizationPassOptions& options);
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELLEGALIZATION_H
