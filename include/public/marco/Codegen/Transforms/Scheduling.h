#ifndef MARCO_CODEGEN_TRANSFORMS_SCHEDULING_H
#define MARCO_CODEGEN_TRANSFORMS_SCHEDULING_H

#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_SCHEDULINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  /// Create a pass performing the scheduling process on a matched model.
  std::unique_ptr<mlir::Pass> createSchedulingPass();

  /// Create a pass performing the scheduling process on a matched model.
  std::unique_ptr<mlir::Pass> createSchedulingPass(
      const SchedulingPassOptions& options);
}

#endif // MARCO_CODEGEN_TRANSFORMS_SCHEDULING_H
