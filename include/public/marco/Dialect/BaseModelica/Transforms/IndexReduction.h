#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_INDEX_REDUCTION_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_INDEX_REDUCTION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_INDEXREDUCTIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createIndexReductionPass();
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_INDEX_REDUCTION_H
