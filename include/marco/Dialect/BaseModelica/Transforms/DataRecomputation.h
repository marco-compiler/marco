#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_DATARECOMPUTATION_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_DATARECOMPUTATION_H
#define DATARECOMPUTATION_H_

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_DATARECOMPUTATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createDataRecomputationPass();
} // namespace mlir::bmodelica

#endif
