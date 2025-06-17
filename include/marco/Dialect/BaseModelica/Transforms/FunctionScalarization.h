#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_FUNCTIONSCALARIZATION_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_FUNCTIONSCALARIZATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_FUNCTIONSCALARIZATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createFunctionScalarizationPass();

std::unique_ptr<mlir::Pass> createFunctionScalarizationPass(
    const FunctionScalarizationPassOptions &options);
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_FUNCTIONSCALARIZATION_H
