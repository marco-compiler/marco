#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_DERIVATIVECHAINRULE_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_DERIVATIVECHAINRULE_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_DERIVATIVECHAINRULEPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createDerivativeChainRulePass();
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_DERIVATIVECHAINRULE_H
