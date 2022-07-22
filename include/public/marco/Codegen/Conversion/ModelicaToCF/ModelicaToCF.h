#ifndef MARCO_CODEGEN_CONVERSION_MODELICATOCF_MODELICATOCF_H
#define MARCO_CODEGEN_CONVERSION_MODELICATOCF_MODELICATOCF_H

#include "mlir/Pass/Pass.h"
#include "llvm/IR/DataLayout.h"

namespace marco::codegen
{
  struct ModelicaToCFOptions
  {
    unsigned int bitWidth = 64;
    bool outputArraysPromotion = true;
    bool inlining = true;

    llvm::DataLayout dataLayout = llvm::DataLayout("");

    static const ModelicaToCFOptions& getDefaultOptions();
  };

	std::unique_ptr<mlir::Pass> createModelicaToCFPass(
      ModelicaToCFOptions options = ModelicaToCFOptions::getDefaultOptions());
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICATOCF_MODELICATOCF_H
