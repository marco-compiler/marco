#ifndef MARCO_DIALECT_RUNTIME_TRANSFORMS_PASSES_H
#define MARCO_DIALECT_RUNTIME_TRANSFORMS_PASSES_H

#include "marco/Dialect/Runtime/Transforms/HeapFunctionsReplacement.h"

namespace mlir::runtime {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "marco/Dialect/Runtime/Transforms/Passes.h.inc"
} // namespace mlir::runtime

#endif // MARCO_DIALECT_RUNTIME_TRANSFORMS_PASSES_H
