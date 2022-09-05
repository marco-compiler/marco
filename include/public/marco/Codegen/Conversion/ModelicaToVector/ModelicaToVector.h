#ifndef MARCO_CODEGEN_CONVERSION_MODELICATOVECTOR_MODELICATOVECTOR_H
#define MARCO_CODEGEN_CONVERSION_MODELICATOVECTOR_MODELICATOVECTOR_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/DataLayout.h"

namespace mlir
{
#define GEN_PASS_DECL_MODELICATOVECTORCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createModelicaToVectorConversionPass();

  std::unique_ptr<mlir::Pass> createModelicaToVectorConversionPass(const ModelicaToVectorConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICATOVECTOR_MODELICATOVECTOR_H
