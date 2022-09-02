#ifndef MARCO_CODEGEN_TRANSFORMS_MATCHING_H
#define MARCO_CODEGEN_TRANSFORMS_MATCHING_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_MATCHINGTESTPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  /// Create a pass performing the matching process on a model.
  /// The pass is intended to be used only for debugging purpose.
  std::unique_ptr<mlir::Pass> createMatchingTestPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_MATCHING_H
