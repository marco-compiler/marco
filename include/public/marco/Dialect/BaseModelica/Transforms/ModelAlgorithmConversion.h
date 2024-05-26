#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELALGORITHMCONVERSION_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELALGORITHMCONVERSION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_MODELALGORITHMCONVERSIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createModelAlgorithmConversionPass();
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELALGORITHMCONVERSION_H
