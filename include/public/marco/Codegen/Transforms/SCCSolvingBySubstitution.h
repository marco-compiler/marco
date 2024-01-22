#ifndef MARCO_CODEGEN_TRANSFORMS_SCCSOLVINGBYSUBSTITUTION_H
#define MARCO_CODEGEN_TRANSFORMS_SCCSOLVINGBYSUBSTITUTION_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_SCCSOLVINGBYSUBSTITUTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createSCCSolvingBySubstitutionPass();

  std::unique_ptr<mlir::Pass> createSCCSolvingBySubstitutionPass(
      const SCCSolvingBySubstitutionPassOptions& options);
}

#endif // MARCO_CODEGEN_TRANSFORMS_SCCSOLVINGBYSUBSTITUTION_H
