#ifndef MARCO_CODEGEN_CONVERSION_IDATOLLVM_IDATOLLVM_H
#define MARCO_CODEGEN_CONVERSION_IDATOLLVM_IDATOLLVM_H

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/DataLayout.h"

namespace mlir
{
#define GEN_PASS_DECL_IDATOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createIDAToLLVMConversionPass();
}

#endif // MARCO_CODEGEN_CONVERSION_IDATOLLVM_IDATOLLVM_H
