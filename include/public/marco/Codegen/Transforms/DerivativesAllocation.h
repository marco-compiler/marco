#ifndef MARCO_CODEGEN_TRANSFORMS_DERIVATIVESALLOCATION_H
#define MARCO_CODEGEN_TRANSFORMS_DERIVATIVESALLOCATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_DERIVATIVESALLOCATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createDerivativesAllocationPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_DERIVATIVESALLOCATION_H
