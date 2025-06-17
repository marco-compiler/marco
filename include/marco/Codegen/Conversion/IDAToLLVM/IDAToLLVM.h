#ifndef MARCO_CODEGEN_CONVERSION_IDATOLLVM_IDATOLLVM_H
#define MARCO_CODEGEN_CONVERSION_IDATOLLVM_IDATOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/DataLayout.h"

namespace mlir {
#define GEN_PASS_DECL_IDATOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

void populateIDAToLLVMConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &typeConverter,
    mlir::SymbolTableCollection &symbolTables);

std::unique_ptr<mlir::Pass> createIDAToLLVMConversionPass();
} // namespace mlir

#endif // MARCO_CODEGEN_CONVERSION_IDATOLLVM_IDATOLLVM_H
