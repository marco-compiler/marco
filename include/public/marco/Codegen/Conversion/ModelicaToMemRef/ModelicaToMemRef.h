#ifndef MARCO_CODEGEN_CONVERSION_MODELICATOLLVM_MODELICATOMEMREF_H
#define MARCO_CODEGEN_CONVERSION_MODELICATOLLVM_MODELICATOMEMREF_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/DataLayout.h"

namespace marco::codegen
{
  struct ModelicaToMemRefOptions
  {
    unsigned int bitWidth = 64;
    bool assertions = true;

    llvm::DataLayout dataLayout = llvm::DataLayout("");

    static const ModelicaToMemRefOptions& getDefaultOptions();
  };

  void populateModelicaToMemRefPatterns(
      mlir::RewritePatternSet& patterns,
      mlir::MLIRContext* context,
      mlir::TypeConverter& typeConverter,
      ModelicaToMemRefOptions options);

  std::unique_ptr<mlir::Pass> createModelicaToMemRefPass(
      ModelicaToMemRefOptions options = ModelicaToMemRefOptions::getDefaultOptions());
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICATOLLVM_MODELICATOMEMREF_H
