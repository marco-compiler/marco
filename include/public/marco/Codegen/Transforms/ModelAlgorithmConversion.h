#ifndef MARCO_CODEGEN_TRANSFORMS_MODELALGORITHMCONVERSION_H
#define MARCO_CODEGEN_TRANSFORMS_MODELALGORITHMCONVERSION_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_MODELALGORITHMCONVERSIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createModelAlgorithmConversionPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELALGORITHMCONVERSION_H
