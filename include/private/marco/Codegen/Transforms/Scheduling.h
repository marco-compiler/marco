#ifndef MARCO_CODGEN_TRANSFORMS_SCHEDULING_H
#define MARCO_CODGEN_TRANSFORMS_SCHEDULING_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createSchedulingPass();

  inline void registerSchedulingPass()
  {
    mlir::registerPass(
        "scheduling", "Perform the scheduling on the model",
        []() -> std::unique_ptr<::mlir::Pass> {
          return createSchedulingPass();
        });
  }
}

#endif // MARCO_CODGEN_TRANSFORMS_SCHEDULING_H
