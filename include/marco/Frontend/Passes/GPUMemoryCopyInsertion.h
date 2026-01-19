#ifndef MARCO_FRONTEND_PASSES_GPUMEMORYCOPYINSERTION_H
#define MARCO_FRONTEND_PASSES_GPUMEMORYCOPYINSERTION_H

#include "mlir/Pass/Pass.h"

namespace marco::frontend {
#define GEN_PASS_DECL_GPUMEMORYCOPYINSERTIONPASS
#include "marco/Frontend/Passes.h.inc"

std::unique_ptr<mlir::Pass> createGPUMemoryCopyInsertionPass();
} // namespace marco::frontend

#endif // MARCO_FRONTEND_PASSES_GPUMEMORYCOPYINSERTION_H
