#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_BINDINGEQUATIONCONVERSION_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_BINDINGEQUATIONCONVERSION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_BINDINGEQUATIONCONVERSIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createBindingEquationConversionPass();
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_BINDINGEQUATIONCONVERSION_H
