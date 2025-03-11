#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATOFUNC_BASEMODELICATOFUNC_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATOFUNC_BASEMODELICATOFUNC_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DECL_BASEMODELICATOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

void populateBaseModelicaToFuncConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter,
    mlir::SymbolTableCollection &symbolTableCollection);

std::unique_ptr<mlir::Pass> createBaseModelicaToFuncConversionPass();

#define GEN_PASS_DECL_BASEMODELICARAWVARIABLESCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

void populateBaseModelicaRawVariablesTypeLegalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter);

void populateBaseModelicaRawVariablesConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context);

std::unique_ptr<mlir::Pass> createBaseModelicaRawVariablesConversionPass();
} // namespace mlir

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATOFUNC_BASEMODELICATOFUNC_H
