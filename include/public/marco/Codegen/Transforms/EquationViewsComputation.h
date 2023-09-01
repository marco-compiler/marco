#ifndef MARCO_CODEGEN_TRANSFORMS_EQUATIONVIEWSCOMPUTATION_H
#define MARCO_CODEGEN_TRANSFORMS_EQUATIONVIEWSCOMPUTATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_EQUATIONVIEWSCOMPUTATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createEquationViewsComputationPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_EQUATIONVIEWSCOMPUTATION_H
