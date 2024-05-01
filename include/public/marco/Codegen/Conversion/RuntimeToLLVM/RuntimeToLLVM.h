#ifndef MARCO_CODEGEN_CONVERSION_RUNTIMETOFUNC_RUNTIMETOLLVM_H
#define MARCO_CODEGEN_CONVERSION_RUNTIMETOFUNC_RUNTIMETOLLVM_H

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DECL_RUNTIMETOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createRuntimeToLLVMConversionPass();

  std::unique_ptr<mlir::Pass> createRuntimeToLLVMConversionPass(
      const RuntimeToLLVMConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_RUNTIMETOFUNC_RUNTIMETOLLVM_H
