#ifndef MARCO_CODEGEN_CONVERSION_MODELICATOMEMREF_MODELICATOMEMREF_H
#define MARCO_CODEGEN_CONVERSION_MODELICATOMEMREF_MODELICATOMEMREF_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/DataLayout.h"

namespace mlir
{
#define GEN_PASS_DECL_MODELICATOMEMREFCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createModelicaToMemRefConversionPass();

  std::unique_ptr<mlir::Pass> createModelicaToMemRefConversionPass(const ModelicaToMemRefConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICATOMEMREF_MODELICATOMEMREF_H
