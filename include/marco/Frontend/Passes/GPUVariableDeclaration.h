#ifndef MARCO_FRONTEND_PASSES_GPUVARIABLEDECLARATION_H
#define MARCO_FRONTEND_PASSES_GPUVARIABLEDECLARATION_H

#include "mlir/Pass/Pass.h"

namespace marco::frontend {
#define GEN_PASS_DECL_GPUVARIABLEDECLARATIONPASS
#include "marco/Frontend/Passes.h.inc"

std::unique_ptr<mlir::Pass> createGPUVariableDeclarationPass();
} // namespace marco::frontend

#endif // MARCO_FRONTEND_PASSES_GPUVARIABLEDECLARATION_H
