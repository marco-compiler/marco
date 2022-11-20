#ifndef MARCO_CODEGEN_TRANSFORMS_DEBUGCANONICALIZER_H

#include "marco/Codegen/Transforms/ModelSolving/Solver.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_DEBUGCANONICALIZERPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createDebugCanonicalizerPass();

  std::unique_ptr<mlir::Pass> createDebugCanonicalizerPass(const DebugCanonicalizerPassOptions& options);
}

#endif // MARCO_CODEGEN_TRANSFORMS_DEBUGCANONICALIZER_H
