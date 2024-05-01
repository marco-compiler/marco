#ifndef MARCO_CODEGEN_TRANSFORMS_FUNCTIONDEFAULTVALUESCONVERSION_H
#define MARCO_CODEGEN_TRANSFORMS_FUNCTIONDEFAULTVALUESCONVERSION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_FUNCTIONDEFAULTVALUESCONVERSIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createFunctionDefaultValuesConversionPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_FUNCTIONDEFAULTVALUESCONVERSION_H
