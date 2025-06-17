#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONFUNCTIONLOOPHOISTING_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONFUNCTIONLOOPHOISTING_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_EQUATIONFUNCTIONLOOPHOISTINGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createEquationFunctionLoopHoistingPass();
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONFUNCTIONLOOPHOISTING_H
