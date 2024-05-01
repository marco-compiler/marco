#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATOVECTOR_BASEMODELICATOVECTOR_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATOVECTOR_BASEMODELICATOVECTOR_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/DataLayout.h"

namespace mlir
{
#define GEN_PASS_DECL_BASEMODELICATOVECTORCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createBaseModelicaToVectorConversionPass();

  std::unique_ptr<mlir::Pass> createBaseModelicaToVectorConversionPass(
      const BaseModelicaToVectorConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATOVECTOR_BASEMODELICATOVECTOR_H
