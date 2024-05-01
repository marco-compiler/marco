#ifndef MARCO_CODEGEN_TRANSFORMS_ARRAYDEALLOCATION_H
#define MARCO_CODEGEN_TRANSFORMS_ARRAYDEALLOCATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_ARRAYDEALLOCATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createArrayDeallocationPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_ARRAYDEALLOCATION_H
