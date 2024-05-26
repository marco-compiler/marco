#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_AUTOMATICDIFFERENTIATION_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_AUTOMATICDIFFERENTIATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_AUTOMATICDIFFERENTIATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createAutomaticDifferentiationPass();
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_AUTOMATICDIFFERENTIATION_H
