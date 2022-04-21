#ifndef MARCO_CODGEN_TRANSFORMS_MATCHING_H
#define MARCO_CODGEN_TRANSFORMS_MATCHING_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
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
