#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SCHEDULEPARALLELIZATION_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SCHEDULEPARALLELIZATION_H

#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::bmodelica {
#define GEN_PASS_DECL_SCHEDULEPARALLELIZATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createScheduleParallelizationPass();

std::unique_ptr<mlir::Pass> createScheduleParallelizationPass(
    const ScheduleParallelizationPassOptions &options);
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SCHEDULEPARALLELIZATION_H
