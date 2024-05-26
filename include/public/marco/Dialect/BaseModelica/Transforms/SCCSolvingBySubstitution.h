#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SCCSOLVINGBYSUBSTITUTION_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SCCSOLVINGBYSUBSTITUTION_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_SCCSOLVINGBYSUBSTITUTIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createSCCSolvingBySubstitutionPass();

  std::unique_ptr<mlir::Pass> createSCCSolvingBySubstitutionPass(
      const SCCSolvingBySubstitutionPassOptions& options);
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SCCSOLVINGBYSUBSTITUTION_H
