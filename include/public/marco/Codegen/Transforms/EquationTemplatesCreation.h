#ifndef MARCO_CODEGEN_TRANSFORMS_EQUATIONTEMPLATESCREATION_H
#define MARCO_CODEGEN_TRANSFORMS_EQUATIONTEMPLATESCREATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_EQUATIONTEMPLATESCREATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createEquationTemplatesCreationPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_EQUATIONTEMPLATESCREATION_H
