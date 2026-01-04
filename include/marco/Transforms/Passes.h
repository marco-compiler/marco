#ifndef MARCO_TRANSFORMS_PASSES_H
#define MARCO_TRANSFORMS_PASSES_H

#include "marco/Transforms/DataRecomputation.h"

namespace mlir {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "marco/Transforms/Passes.h.inc"
} // namespace mlir

#endif // MARCO_TRANSFORMS_PASSES_H
