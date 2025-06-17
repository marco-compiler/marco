#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_FUNCTIONDEFAULTVALUESCONVERSION_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_FUNCTIONDEFAULTVALUESCONVERSION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_FUNCTIONDEFAULTVALUESCONVERSIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createFunctionDefaultValuesConversionPass();
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_FUNCTIONDEFAULTVALUESCONVERSION_H
