#ifndef MARCO_CODEGEN_CONVERSION_RUNTIMETOLLVM_RUNTIMETOLLVM_H
#define MARCO_CODEGEN_CONVERSION_RUNTIMETOLLVM_RUNTIMETOLLVM_H

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DECL_RUNTIMETOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  void populateRuntimeToLLVMPatterns(
      mlir::RewritePatternSet& patterns,
      mlir::LLVMTypeConverter& typeConverter,
      mlir::SymbolTableCollection& symbolTableCollection);

  std::unique_ptr<mlir::Pass> createRuntimeToLLVMConversionPass();
}

#endif // MARCO_CODEGEN_CONVERSION_RUNTIMETOLLVM_RUNTIMETOLLVM_H
