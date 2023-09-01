#ifndef MARCO_CODEGEN_TRANSFORMS_VARIABLESPROMOTION_H
#define MARCO_CODEGEN_TRANSFORMS_VARIABLESPROMOTION_H

#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_VARIABLESPROMOTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createVariablesPromotionPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_VARIABLESPROMOTION_H
