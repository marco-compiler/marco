#ifndef MARCO_CODEGEN_CONVERSION_IDATOFUNC_IDATOFUNC_H
#define MARCO_CODEGEN_CONVERSION_IDATOFUNC_IDATOFUNC_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/DataLayout.h"

namespace mlir {
#define GEN_PASS_DECL_IDATOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createIDAToFuncConversionPass();
} // namespace mlir

#endif // MARCO_CODEGEN_CONVERSION_IDATOFUNC_IDATOFUNC_H
