#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SCCSOLVINGWITHKINSOL_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SCCSOLVINGWITHKINSOL_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_SCCSOLVINGWITHKINSOLPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createSCCSolvingWithKINSOLPass();

  std::unique_ptr<mlir::Pass> createSCCSolvingWithKINSOLPass(
      const SCCSolvingWithKINSOLPassOptions& options);
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SCCSOLVINGWITHKINSOL_H
