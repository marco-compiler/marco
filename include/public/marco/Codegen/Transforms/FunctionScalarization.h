#ifndef MARCO_CODEGEN_TRANSFORMS_FUNCTIONSCALARIZATION_H
#define MARCO_CODEGEN_TRANSFORMS_FUNCTIONSCALARIZATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_FUNCTIONSCALARIZATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createFunctionScalarizationPass();

  std::unique_ptr<mlir::Pass> createFunctionScalarizationPass(const FunctionScalarizationPassOptions& options);
}

#endif // MARCO_CODEGEN_TRANSFORMS_FUNCTIONSCALARIZATION_H
