#ifndef MARCO_CODGEN_TRANSFORMS_SCHEDULING_H
#define MARCO_CODGEN_TRANSFORMS_SCHEDULING_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
  /// Create a pass performing the scheduling process on a matched model.
  /// The pass is intended to be used only for debugging purpose.
  std::unique_ptr<mlir::Pass> createSchedulingPass();
}

#endif // MARCO_CODGEN_TRANSFORMS_SCHEDULING_H
