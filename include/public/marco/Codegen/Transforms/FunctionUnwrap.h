#ifndef MARCO_CODEGEN_TRANSFORMS_FUNCTIONUNWRAP_H
#define MARCO_CODEGEN_TRANSFORMS_FUNCTIONUNWRAP_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_FUNCTIONUNWRAPPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createFunctionUnwrapPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_FUNCTIONUNWRAP_H
