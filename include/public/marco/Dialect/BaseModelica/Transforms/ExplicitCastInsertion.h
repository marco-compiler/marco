#ifndef MARCO_DIALECT_MDOELICA_TRANSFORMS_EXPLICITCASTINSERTION_H
#define MARCO_DIALECT_MDOELICA_TRANSFORMS_EXPLICITCASTINSERTION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_EXPLICITCASTINSERTIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createExplicitCastInsertionPass();
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_MDOELICA_TRANSFORMS_EXPLICITCASTINSERTION_H
