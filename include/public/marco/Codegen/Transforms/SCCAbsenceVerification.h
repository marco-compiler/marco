#ifndef MARCO_CODEGEN_TRANSFORMS_SCCABSENCEVERIFICATION_H
#define MARCO_CODEGEN_TRANSFORMS_SCCABSENCEVERIFICATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_SCCABSENCEVERIFICATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createSCCAbsenceVerificationPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_SCCABSENCEVERIFICATION_H
