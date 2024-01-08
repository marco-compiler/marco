#ifndef MARCO_CODEGEN_TRANSFORMS_EQUATIONEXPLICITATION_H
#define MARCO_CODEGEN_TRANSFORMS_EQUATIONEXPLICITATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_EQUATIONEXPLICITATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createEquationExplicitationPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_EQUATIONEXPLICITATION_H
