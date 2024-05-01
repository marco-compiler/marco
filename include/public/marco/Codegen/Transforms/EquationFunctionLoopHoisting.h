#ifndef MARCO_CODEGEN_TRANSFORMS_EQUATIONFUNCTIONLOOPHOISTING_H
#define MARCO_CODEGEN_TRANSFORMS_EQUATIONFUNCTIONLOOPHOISTING_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_EQUATIONFUNCTIONLOOPHOISTINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createEquationFunctionLoopHoistingPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_EQUATIONFUNCTIONLOOPHOISTING_H
