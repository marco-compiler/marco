#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATOARITH_BASEMODELICATOARITH_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATOARITH_BASEMODELICATOARITH_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DECL_BASEMODELICATOARITHCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

void populateBaseModelicaToArithConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter);

std::unique_ptr<mlir::Pass> createBaseModelicaToArithConversionPass();
} // namespace mlir

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATOARITH_BASEMODELICATOARITH_H
