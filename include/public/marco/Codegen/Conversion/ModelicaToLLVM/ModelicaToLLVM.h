#ifndef MARCO_CODEGEN_CONVERSION_MODELICATOLLVM_MODELICATOLLVM_H
#define MARCO_CODEGEN_CONVERSION_MODELICATOLLVM_MODELICATOLLVM_H

#include "mlir/Pass/Pass.h"
#include "llvm/IR/DataLayout.h"

namespace mlir
{
#define GEN_PASS_DECL_MODELICATOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createModelicaToLLVMConversionPass();

  std::unique_ptr<mlir::Pass> createModelicaToLLVMConversionPass(const ModelicaToLLVMConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICATOLLVM_MODELICATOLLVM_H
