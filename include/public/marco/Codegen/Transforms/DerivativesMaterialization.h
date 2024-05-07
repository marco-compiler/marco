#ifndef MARCO_CODEGEN_TRANSFORMS_DERIVATIVESMATERIALIZATION_H
#define MARCO_CODEGEN_TRANSFORMS_DERIVATIVESMATERIALIZATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_DERIVATIVESMATERIALIZATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createDerivativesMaterializationPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_DERIVATIVESMATERIALIZATION_H