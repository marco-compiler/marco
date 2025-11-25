#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONOFFLOADINGATTACHTARGETS_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONOFFLOADINGATTACHTARGETS_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_EQUATIONOFFLOADINGATTACHTARGETSPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createEquationOffloadingAttachTargetsPass();

std::unique_ptr<mlir::Pass> createEquationOffloadingAttachTargetsPass(
    const EquationOffloadingAttachTargetsPassOptions &options);
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONOFFLOADINGATTACHTARGETS_H
