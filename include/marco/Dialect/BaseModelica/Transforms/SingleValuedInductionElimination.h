#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SINGLEVALUEDINDUCTIONELIMINATION_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SINGLEVALUEDINDUCTIONELIMINATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_SINGLEVALUEDINDUCTIONELIMINATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createSingleValuedInductionEliminationPass();
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SINGLEVALUEDINDUCTIONELIMINATION_H
