#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATORUNTIMECALL_BASEMODELICATORUNTIMECALL_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATORUNTIMECALL_BASEMODELICATORUNTIMECALL_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DECL_BASEMODELICATORUNTIMECALLCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

void populateBaseModelicaToRuntimeCallConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter,
    mlir::SymbolTableCollection &symbolTableCollection);

std::unique_ptr<mlir::Pass> createBaseModelicaToRuntimeCallConversionPass();
} // namespace mlir

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATORUNTIMECALL_BASEMODELICATORUNTIMECALL_H
