#ifndef MARCO_FRONTEND_PASSES_HEAPFUNCTIONSREPLACEMENT_H
#define MARCO_FRONTEND_PASSES_HEAPFUNCTIONSREPLACEMENT_H

#include "mlir/Pass/Pass.h"

namespace marco::frontend {
#define GEN_PASS_DECL_HEAPFUNCTIONSREPLACEMENTPASS
#include "marco/Frontend/Passes.h.inc"

std::unique_ptr<mlir::Pass> createHeapFunctionsReplacementPass();
} // namespace marco::frontend

#endif // MARCO_FRONTEND_PASSES_HEAPFUNCTIONSREPLACEMENT_H
