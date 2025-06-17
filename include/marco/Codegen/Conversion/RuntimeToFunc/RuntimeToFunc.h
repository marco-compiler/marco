#ifndef MARCO_CODEGEN_CONVERSION_RUNTIMETOFUNC_RUNTIMETOFUNC_H
#define MARCO_CODEGEN_CONVERSION_RUNTIMETOFUNC_RUNTIMETOFUNC_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DECL_RUNTIMETOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createRuntimeToFuncConversionPass();
} // namespace mlir

#endif // MARCO_CODEGEN_CONVERSION_RUNTIMETOFUNC_RUNTIMETOFUNC_H
