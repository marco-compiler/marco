#ifndef MARCO_CODEGEN_TRANSFORMS_BINDINGEQUATIONCONVERSION_H
#define MARCO_CODEGEN_TRANSFORMS_BINDINGEQUATIONCONVERSION_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_BINDINGEQUATIONCONVERSIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createBindingEquationConversionPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_BINDINGEQUATIONCONVERSION_H