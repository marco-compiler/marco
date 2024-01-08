#ifndef MARCO_CODEGEN_TRANSFORMS_SCCSOLVING_H
#define MARCO_CODEGEN_TRANSFORMS_SCCSOLVING_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_SCCSOLVINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createSCCSolvingPass();

  std::unique_ptr<mlir::Pass> createSCCSolvingPass(
      const SCCSolvingPassOptions& options);
}

#endif // MARCO_CODEGEN_TRANSFORMS_SCCSOLVING_H
