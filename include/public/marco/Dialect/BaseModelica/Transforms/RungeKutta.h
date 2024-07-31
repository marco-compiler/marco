#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_RUNGEKUTTA_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_RUNGEKUTTA_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_RUNGEKUTTAPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createRungeKuttaPass();

std::unique_ptr<mlir::Pass>
createRungeKuttaPass(const RungeKuttaPassOptions &options);
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_RUNGEKUTTA_H
