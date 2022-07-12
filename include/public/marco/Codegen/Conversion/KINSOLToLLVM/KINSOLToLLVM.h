#ifndef MARCO_CODEGEN_CONVERSION_KINSOL_KINSOLTOLLVM_H
#define MARCO_CODEGEN_CONVERSION_KINSOL_KINSOLTOLLVM_H

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/DataLayout.h"

namespace mlir
{
#define GEN_PASS_DECL_KINSOLTOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  void populateKINSOLStructuralTypeConversionsAndLegality(
      mlir::LLVMTypeConverter& typeConverter,
      mlir::RewritePatternSet& patterns,
      mlir::ConversionTarget& target);

  std::unique_ptr<mlir::Pass> createKINSOLToLLVMConversionPass();

  std::unique_ptr<mlir::Pass> createKINSOLToLLVMConversionPass(const KINSOLToLLVMConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_KINSOL_KINSOLTOLLVM_H
