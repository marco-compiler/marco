#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATOFUNC_BASEMODELICATOFUNC_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATOFUNC_BASEMODELICATOFUNC_H

#include "marco/Codegen/Conversion/BaseModelicaCommon/CTypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DECL_BASEMODELICATOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

void populateBaseModelicaToFuncConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter,
    mlir::SymbolTableCollection &symbolTables);

void populateBaseModelicaExternalCallConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    bmodelica::CTypeConverter &CTypeConverter,
    mlir::SymbolTableCollection &symbolTables);

std::unique_ptr<mlir::Pass> createBaseModelicaToFuncConversionPass();

#define GEN_PASS_DECL_BASEMODELICARAWVARIABLESCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

void populateBaseModelicaRawVariablesTypeLegalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter);

std::unique_ptr<mlir::Pass> createBaseModelicaRawVariablesConversionPass();
} // namespace mlir

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATOFUNC_BASEMODELICATOFUNC_H
