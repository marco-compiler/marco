#ifndef MARCO_CODEGEN_CONVERSION_MODELICATOCF_MODELICATOCF_H
#define MARCO_CODEGEN_CONVERSION_MODELICATOCF_MODELICATOCF_H

#include "mlir/Pass/Pass.h"
#include "llvm/IR/DataLayout.h"

namespace mlir
{
#define GEN_PASS_DECL_MODELICATOCFCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createModelicaToCFConversionPass();

  std::unique_ptr<mlir::Pass> createModelicaToCFConversionPass(const ModelicaToCFConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICATOCF_MODELICATOCF_H
