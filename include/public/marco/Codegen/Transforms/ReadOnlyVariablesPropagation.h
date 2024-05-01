#ifndef MARCO_CODEGEN_TRANSFORMS_READONLYVARIABLESPROPAGATION_H
#define MARCO_CODEGEN_TRANSFORMS_READONLYVARIABLESPROPAGATION_H

#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_READONLYVARIABLESPROPAGATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createReadOnlyVariablesPropagationPass();

  std::unique_ptr<mlir::Pass> createReadOnlyVariablesPropagationPass(
      const ReadOnlyVariablesPropagationPassOptions& options);
}

#endif // MARCO_CODEGEN_TRANSFORMS_READONLYVARIABLESPROPAGATION_H
