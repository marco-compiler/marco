#ifndef MARCO_CODEGEN_TRANSFORMS_SCHEDULEPARALLELIZATION_H
#define MARCO_CODEGEN_TRANSFORMS_SCHEDULEPARALLELIZATION_H

#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_SCHEDULEPARALLELIZATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createScheduleParallelizationPass();

  std::unique_ptr<mlir::Pass> createScheduleParallelizationPass(
      const ScheduleParallelizationPassOptions& options);
}

#endif // MARCO_CODEGEN_TRANSFORMS_SCHEDULEPARALLELIZATION_H
