#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATOLLVM_BASEMODELICATOLLVM_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATOLLVM_BASEMODELICATOLLVM_H

#include "mlir/Pass/Pass.h"
#include "llvm/IR/DataLayout.h"

namespace mlir
{
#define GEN_PASS_DECL_BASEMODELICATOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createBaseModelicaToLLVMConversionPass();

  std::unique_ptr<mlir::Pass> createBaseModelicaToLLVMConversionPass(
      const BaseModelicaToLLVMConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATOLLVM_BASEMODELICATOLLVM_H
