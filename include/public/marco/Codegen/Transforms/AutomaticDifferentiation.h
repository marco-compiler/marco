#ifndef MARCO_CODEGEN_TRANSFORMS_AUTOMATICDIFFERENTIATION_H
#define MARCO_CODEGEN_TRANSFORMS_AUTOMATICDIFFERENTIATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_AUTOMATICDIFFERENTIATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createAutomaticDifferentiationPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_AUTOMATICDIFFERENTIATION_H
