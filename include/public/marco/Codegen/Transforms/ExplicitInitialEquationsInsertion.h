#ifndef MARCO_CODEGEN_TRANSFORMS_EXPLICITINITIALEQUATIONSINSERTION_H
#define MARCO_CODEGEN_TRANSFORMS_EXPLICITINITIALEQUATIONSINSERTION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_EXPLICITINITIALEQUATIONSINSERTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createExplicitInitialEquationsInsertionPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_EXPLICITINITIALEQUATIONSINSERTION_H
