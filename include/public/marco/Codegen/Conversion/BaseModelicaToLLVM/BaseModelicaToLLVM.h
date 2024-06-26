#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATOLLVM_BASEMODELICATOLLVM_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATOLLVM_BASEMODELICATOLLVM_H

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir
{
#define GEN_PASS_DECL_BASEMODELICATOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  void populateBaseModelicaToLLVMConversionPatterns(
      mlir::RewritePatternSet& patterns,
      mlir::LLVMTypeConverter& typeConverter);

  std::unique_ptr<mlir::Pass> createBaseModelicaToLLVMConversionPass();

  void registerConvertBaseModelicaToLLVMInterface(
      mlir::DialectRegistry& registry);
}

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATOLLVM_BASEMODELICATOLLVM_H
