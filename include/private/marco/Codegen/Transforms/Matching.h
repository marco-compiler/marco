#ifndef MARCO_CODGEN_TRANSFORMS_MATCHING_H
#define MARCO_CODGEN_TRANSFORMS_MATCHING_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
  /// Create a pass performing the matching process on a model.
  /// The pass is intended to be used only for debugging purpose.
  std::unique_ptr<mlir::Pass> createMatchingPass();

  inline void registerMatchingPass()
  {
    mlir::registerPass(
        "matching", "Perform the matching on the model",
        []() -> std::unique_ptr<::mlir::Pass> {
          return createMatchingPass();
        });
  }
}

#endif // MARCO_CODGEN_TRANSFORMS_MATCHING_H
