#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_INITIALCONDITIONSSOLVING_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_INITIALCONDITIONSSOLVING_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_INITIALCONDITIONSSOLVINGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createInitialConditionsSolvingPass();
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_INITIALCONDITIONSSOLVING_H
