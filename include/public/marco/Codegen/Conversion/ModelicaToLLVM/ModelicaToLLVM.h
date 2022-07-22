#ifndef MARCO_CODEGEN_CONVERSION_MODELICATOLLVM_MODELICATOLLVM_H
#define MARCO_CODEGEN_CONVERSION_MODELICATOLLVM_MODELICATOLLVM_H

#include "mlir/Pass/Pass.h"
#include "llvm/IR/DataLayout.h"

namespace marco::codegen
{
	struct ModelicaToLLVMOptions
	{
    unsigned int bitWidth = 64;
		bool assertions = true;

    llvm::DataLayout dataLayout = llvm::DataLayout("");

		static const ModelicaToLLVMOptions& getDefaultOptions();
	};

	std::unique_ptr<mlir::Pass> createModelicaToLLVMPass(
      ModelicaToLLVMOptions options = ModelicaToLLVMOptions::getDefaultOptions());
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICATOLLVM_MODELICATOLLVM_H
