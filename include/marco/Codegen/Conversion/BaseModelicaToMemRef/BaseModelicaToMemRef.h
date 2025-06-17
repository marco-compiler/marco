#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATOMEMREF_BASEMODELICATOMEMREF_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATOMEMREF_BASEMODELICATOMEMREF_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DECL_BASEMODELICATOMEMREFCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

void populateBaseModelicaToMemRefConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter,
    mlir::SymbolTableCollection &symbolTableCollection);

std::unique_ptr<mlir::Pass> createBaseModelicaToMemRefConversionPass();

std::unique_ptr<mlir::Pass> createBaseModelicaRawVariablesConversionPass();
} // namespace mlir

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATOMEMREF_BASEMODELICATOMEMREF_H
