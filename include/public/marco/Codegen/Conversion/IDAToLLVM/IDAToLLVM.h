#ifndef MARCO_CODEGEN_CONVERSION_IDA_IDATOLLVM_H
#define MARCO_CODEGEN_CONVERSION_IDA_IDATOLLVM_H

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace marco::codegen
{
  void populateIDAStructuralTypeConversionsAndLegality(
      mlir::LLVMTypeConverter& typeConverter,
      mlir::RewritePatternSet& patterns,
      mlir::ConversionTarget& target);

  std::unique_ptr<mlir::Pass> createIDAToLLVMPass();
}

#endif // MARCO_CODEGEN_CONVERSION_IDA_IDATOLLVM_H
