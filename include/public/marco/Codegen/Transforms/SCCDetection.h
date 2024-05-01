#ifndef MARCO_CODEGEN_TRANSFORMS_SCCDETECTION_H
#define MARCO_CODEGEN_TRANSFORMS_SCCDETECTION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_SCCDETECTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createSCCDetectionPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_SCCDETECTION_H
