#ifndef MARCO_CODEGEN_TRANSFORMS_SCCSOLVINGWITHKINSOL_H
#define MARCO_CODEGEN_TRANSFORMS_SCCSOLVINGWITHKINSOL_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_SCCSOLVINGWITHKINSOLPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createSCCSolvingWithKINSOLPass();

  std::unique_ptr<mlir::Pass> createSCCSolvingWithKINSOLPass(
      const SCCSolvingWithKINSOLPassOptions& options);
}

#endif // MARCO_CODEGEN_TRANSFORMS_SCCSOLVINGWITHKINSOL_H
