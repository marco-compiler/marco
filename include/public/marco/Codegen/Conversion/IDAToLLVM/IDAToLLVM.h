#ifndef MARCO_CODEGEN_CONVERSION_IDATOLLVM_IDATOLLVM_H
#define MARCO_CODEGEN_CONVERSION_IDATOLLVM_IDATOLLVM_H

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/DataLayout.h"

namespace marco::codegen
{
  struct IDAToLLVMOptions
  {
    llvm::DataLayout dataLayout = llvm::DataLayout("");

    static const IDAToLLVMOptions& getDefaultOptions();
  };

  void populateIDAStructuralTypeConversionsAndLegality(
      mlir::LLVMTypeConverter& typeConverter,
      mlir::RewritePatternSet& patterns,
      mlir::ConversionTarget& target);

  std::unique_ptr<mlir::Pass> createIDAToLLVMPass(
      IDAToLLVMOptions options = IDAToLLVMOptions::getDefaultOptions());
}

#endif // MARCO_CODEGEN_CONVERSION_IDATOLLVM_IDATOLLVM_H
