#ifndef MARCO_TRANSFORMS_ARRAYDEALLOCATIONPASS_H
#define MARCO_TRANSFORMS_ARRAYDEALLOCATIONPASS_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createArrayDeallocationPass();

  inline void registerArrayDeallocationPass()
  {
    mlir::registerPass(
        "array-deallocation", "Modelica: automatic array deallocation",
        []() -> std::unique_ptr<::mlir::Pass> {
          return createArrayDeallocationPass();
        });
  }
}

#endif // MARCO_TRANSFORMS_ARRAYDEALLOCATIONPASS_H
