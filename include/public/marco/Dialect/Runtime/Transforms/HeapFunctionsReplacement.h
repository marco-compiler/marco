#ifndef MARCO_DIALECT_RUNTIME_TRANSFORMS_HEAPFUNCTIONSREPLACEMENT_H
#define MARCO_DIALECT_RUNTIME_TRANSFORMS_HEAPFUNCTIONSREPLACEMENT_H

#include "mlir/Pass/Pass.h"

namespace mlir::runtime
{
#define GEN_PASS_DECL_HEAPFUNCTIONSREPLACEMENTPASS
#include "marco/Dialect/Runtime/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createHeapFunctionsReplacementPass();
}

#endif // MARCO_DIALECT_RUNTIME_TRANSFORMS_HEAPFUNCTIONSREPLACEMENT_H
