#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_RANGEBOUNDARIESINFERENCE_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_RANGEBOUNDARIESINFERENCE_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_RANGEBOUNDARIESINFERENCEPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createRangeBoundariesInferencePass();
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_RANGEBOUNDARIESINFERENCE_H
