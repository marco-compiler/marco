#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATOMEMREF_BASEMODELICATOMEMREF_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATOMEMREF_BASEMODELICATOMEMREF_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/DataLayout.h"

namespace mlir
{
#define GEN_PASS_DECL_BASEMODELICATOMEMREFCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createBaseModelicaToMemRefConversionPass();

  std::unique_ptr<mlir::Pass> createBaseModelicaToMemRefConversionPass(
      const BaseModelicaToMemRefConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATOMEMREF_BASEMODELICATOMEMREF_H
