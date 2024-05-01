#ifndef MARCO_CODEGEN_TRANSFORMS_EXPLICSTARTVALUEINSERTION_H
#define MARCO_CODEGEN_TRANSFORMS_EXPLICSTARTVALUEINSERTION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_EXPLICITSTARTVALUEINSERTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createExplicitStartValueInsertionPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_EXPLICSTARTVALUEINSERTION_H
