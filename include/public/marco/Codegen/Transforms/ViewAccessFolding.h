#ifndef MARCO_CODEGEN_TRANSFORMS_VIEWACCESSFOLDING_H
#define MARCO_CODEGEN_TRANSFORMS_VIEWACCESSFOLDING_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_VIEWACCESSFOLDINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createViewAccessFoldingPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_VIEWACCESSFOLDING_H
