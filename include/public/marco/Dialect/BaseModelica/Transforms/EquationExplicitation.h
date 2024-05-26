#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONEXPLICITATION_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONEXPLICITATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_EQUATIONEXPLICITATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createEquationExplicitationPass();
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONEXPLICITATION_H
