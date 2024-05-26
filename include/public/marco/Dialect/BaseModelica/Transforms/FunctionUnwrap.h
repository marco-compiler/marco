#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_FUNCTIONUNWRAP_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_FUNCTIONUNWRAP_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_FUNCTIONUNWRAPPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createFunctionUnwrapPass();
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_FUNCTIONUNWRAP_H
