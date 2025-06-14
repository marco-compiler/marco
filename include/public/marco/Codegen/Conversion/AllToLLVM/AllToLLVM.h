#ifndef MARCO_CODEGEN_CONVERSION_ALLTOLLVM_ALLTOLLVM_H
#define MARCO_CODEGEN_CONVERSION_ALLTOLLVM_ALLTOLLVM_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_ALLTOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createAllToLLVMConversionPass();
} // namespace mlir

#endif // MARCO_CODEGEN_CONVERSION_ALLTOLLVM_ALLTOLLVM_H
