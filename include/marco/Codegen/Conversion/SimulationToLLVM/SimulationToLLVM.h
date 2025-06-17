#ifndef MARCO_CODEGEN_CONVERSION_SIMULATIONTOLLVM_SIMULATIONTOLLVM_H
#define MARCO_CODEGEN_CONVERSION_SIMULATIONTOLLVM_SIMULATIONTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DECL_SIMULATIONTOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

void populateSimulationToLLVMPatterns(
    mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &typeConverter,
    mlir::SymbolTableCollection &symbolTableCollection);

std::unique_ptr<mlir::Pass> createSimulationToLLVMConversionPass();
} // namespace mlir

#endif // MARCO_CODEGEN_CONVERSION_SIMULATIONTOLLVM_SIMULATIONTOLLVM_H
