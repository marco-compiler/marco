#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EULERFORWARD_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EULERFORWARD_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_EULERFORWARDPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createEulerForwardPass();

std::unique_ptr<mlir::Pass>
createEulerForwardPass(const EulerForwardPassOptions &options);
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EULERFORWARD_H
