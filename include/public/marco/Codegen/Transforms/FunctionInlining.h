#ifndef MARCO_CODEGEN_TRANSFORMS_FUNCTIONINLINING_H
#define MARCO_CODEGEN_TRANSFORMS_FUNCTIONINLINING_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_FUNCTIONINLININGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createFunctionInliningPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_FUNCTIONINLINING_H
