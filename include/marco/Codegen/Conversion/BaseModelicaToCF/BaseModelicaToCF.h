#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATOCF_BASEMODELICATOCF_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATOCF_BASEMODELICATOCF_H

#include "mlir/Pass/Pass.h"
#include "llvm/IR/DataLayout.h"

namespace mlir {
#define GEN_PASS_DECL_BASEMODELICATOCFCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createBaseModelicaToCFConversionPass();
} // namespace mlir

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATOCF_BASEMODELICATOCF_H
