#ifndef MARCO_CODEGEN_TRANSFORMS_EQUATIONINDUCTIONSEXPLICITATION_H
#define MARCO_CODEGEN_TRANSFORMS_EQUATIONINDUCTIONSEXPLICITATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_EQUATIONINDUCTIONSEXPLICITATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createEquationInductionsExplicitationPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_EQUATIONINDUCTIONSEXPLICITATION_H
