#ifndef MARCO_CODEGEN_TRANSFORMS_BUFFERIZATION_H
#define MARCO_CODEGEN_TRANSFORMS_BUFFERIZATION_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createBufferizationPass();

  inline void registerBufferDeallocationPass()
  {
    mlir::registerPass(
        "buffer-deallocation", "Modelica: automatic buffer deallocation",
        []() -> std::unique_ptr<::mlir::Pass> {
          return createBufferDeallocationPass();
        });
  }
}

#endif // MARCO_CODEGEN_TRANSFORMS_BUFFERIZATION_H
