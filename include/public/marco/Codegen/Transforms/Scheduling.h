#ifndef MARCO_CODEGEN_TRANSFORMS_SCHEDULING_H
#define MARCO_CODEGEN_TRANSFORMS_SCHEDULING_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_SCHEDULINGPASS
#define GEN_PASS_DECL_SCHEDULINGTESTPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  /// Create a pass performing the scheduling process on a matched model.
  std::unique_ptr<mlir::Pass> createSchedulingPass();

  /// Create a pass performing the scheduling process on a matched model.
  std::unique_ptr<mlir::Pass> createSchedulingPass(const SchedulingPassOptions& options);

  /// Create a pass performing the scheduling process on a matched model.
  /// The pass is intended to be used only for debugging purpose.
  std::unique_ptr<mlir::Pass> createSchedulingTestPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_SCHEDULING_H
