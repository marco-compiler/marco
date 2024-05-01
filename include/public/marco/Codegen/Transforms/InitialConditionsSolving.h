#ifndef MARCO_CODEGEN_TRANSFORMS_INITIALCONDITIONSSOLVING_H
#define MARCO_CODEGEN_TRANSFORMS_INITIALCONDITIONSSOLVING_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_INITIALCONDITIONSSOLVINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createInitialConditionsSolvingPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_INITIALCONDITIONSSOLVING_H
