#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATOTENSOR_BASEMODELICATOTENSOR_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATOTENSOR_BASEMODELICATOTENSOR_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DECL_BASEMODELICATOTENSORCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  void populateBaseModelicaToTensorConversionPatterns(
      mlir::RewritePatternSet& patterns,
      mlir::MLIRContext* context,
      mlir::TypeConverter& typeConverter);

  std::unique_ptr<mlir::Pass> createBaseModelicaToTensorConversionPass();
}

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATOTENSOR_BASEMODELICATOTENSOR_H
