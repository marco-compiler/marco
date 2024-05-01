#ifndef MARCO_CODEGEN_TRANSFORMS_EQUATIONACCESSSPLIT_H
#define MARCO_CODEGEN_TRANSFORMS_EQUATIONACCESSSPLIT_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_EQUATIONACCESSSPLIT
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createEquationAccessSplitPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_EQUATIONACCESSSPLIT_H
