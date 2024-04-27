#ifndef MARCO_CODEGEN_TRANSFORMS_SINGLEVALUEDINDUCTIONELIMINATION_H
#define MARCO_CODEGEN_TRANSFORMS_SINGLEVALUEDINDUCTIONELIMINATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_SINGLEVALUEDINDUCTIONELIMINATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createSingleValuedInductionEliminationPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_SINGLEVALUEDINDUCTIONELIMINATION_H