#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SCHEDULING_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SCHEDULING_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_SCHEDULINGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

  /// Create a pass performing the scheduling process on a matched model.
  std::unique_ptr<mlir::Pass> createSchedulingPass();
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SCHEDULING_H
