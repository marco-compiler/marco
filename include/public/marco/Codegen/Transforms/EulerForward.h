#ifndef MARCO_CODEGEN_TRANSFORMS_EULERFORWARD_H
#define MARCO_CODEGEN_TRANSFORMS_EULERFORWARD_H

#include "marco/VariableFilter/VariableFilter.h"
#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_EULERFORWARDPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createEulerForwardPass();

  std::unique_ptr<mlir::Pass> createEulerForwardPass(
      const EulerForwardPassOptions& options);
}

#endif // MARCO_CODEGEN_TRANSFORMS_EULERFORWARD_H
