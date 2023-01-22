#ifndef MARCO_CODEGEN_TRANSFORMS_PARAMETERSPROPAGATION_H
#define MARCO_CODEGEN_TRANSFORMS_PARAMETERSPROPAGATION_H

#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_PARAMETERSPROPAGATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createParametersPropagationPass();

  std::unique_ptr<mlir::Pass> createParametersPropagationPass(
      const ParametersPropagationPassOptions& options);
}

#endif // MARCO_CODEGEN_TRANSFORMS_PARAMETERSPROPAGATION_H
