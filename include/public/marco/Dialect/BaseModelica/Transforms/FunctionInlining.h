#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_FUNCTIONINLINING_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_FUNCTIONINLINING_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_FUNCTIONINLININGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createFunctionInliningPass();
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_FUNCTIONINLINING_H
