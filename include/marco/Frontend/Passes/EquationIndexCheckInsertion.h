#ifndef MARCO_FRONTEND_PASSES_EQUATIONINDEXCHECKINSERTION_H
#define MARCO_FRONTEND_PASSES_EQUATIONINDEXCHECKINSERTION_H

#include "mlir/Pass/Pass.h"

namespace marco::frontend {
#define GEN_PASS_DECL_EQUATIONINDEXCHECKINSERTIONPASS
#include "marco/Frontend/Passes.h.inc"

std::unique_ptr<mlir::Pass> createEquationIndexCheckInsertionPass();
} // namespace marco::frontend

#endif // MARCO_FRONTEND_PASSES_EQUATIONINDEXCHECKINSERTION_H
