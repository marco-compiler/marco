#ifndef MARCO_CODEGEN_TRANSFORMS_MATCHING_H
#define MARCO_CODEGEN_TRANSFORMS_MATCHING_H

#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_MATCHINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  /// Create a pass performing the matching process on a model.
  std::unique_ptr<mlir::Pass> createMatchingPass();

  /// Create a pass performing the matching process on a model.
  std::unique_ptr<mlir::Pass> createMatchingPass(
      const MatchingPassOptions& options);
}

#endif // MARCO_CODEGEN_TRANSFORMS_MATCHING_H
