#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATOLINALG_BASEMODELICATOLINALG_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATOLINALG_BASEMODELICATOLINALG_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DECL_BASEMODELICATOLINALGCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  void populateBaseModelicaToLinalgConversionPatterns(
      mlir::RewritePatternSet& patterns,
      mlir::MLIRContext* context,
      mlir::TypeConverter& typeConverter);

  std::unique_ptr<mlir::Pass> createBaseModelicaToLinalgConversionPass();
}

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATOLINALG_BASEMODELICATOLINALG_H
