#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_VARIABLESPROMOTION_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_VARIABLESPROMOTION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_VARIABLESPROMOTIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createVariablesPromotionPass();
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_VARIABLESPROMOTION_H
