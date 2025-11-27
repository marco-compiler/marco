#ifndef MARCO_FRONTEND_PASSES_AGGRESSIVELICM_H
#define MARCO_FRONTEND_PASSES_AGGRESSIVELICM_H

#include "mlir/Pass/Pass.h"

namespace marco::frontend {
#define GEN_PASS_DECL_AGGRESSIVELICMPASS
#include "marco/Frontend/Passes.h.inc"

/// Create a pass that hoists loop-invariant code out of loop bodies.
std::unique_ptr<mlir::Pass> createAggressiveLICMPass();
} // namespace marco::frontend

#endif // MARCO_FRONTEND_PASSES_AGGRESSIVELICM_H
