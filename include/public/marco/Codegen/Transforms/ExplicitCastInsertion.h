#ifndef MARCO_CODEGEN_TRANSFORMS_EXPLICITCASTINSERTION_H
#define MARCO_CODEGEN_TRANSFORMS_EXPLICITCASTINSERTION_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_EXPLICITCASTINSERTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createExplicitCastInsertionPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_EXPLICITCASTINSERTION_H
