#ifndef MARCO_CODEN_CONVERSION_MODELICA_LOWERTOLLVM_H
#define MARCO_CODEN_CONVERSION_MODELICA_LOWERTOLLVM_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
	struct ModelicaToLLVMConversionOptions
	{
		bool assertions = true;

		static const ModelicaToLLVMConversionOptions& getDefaultOptions() {
			static ModelicaToLLVMConversionOptions options;
			return options;
		}
	};

	std::unique_ptr<mlir::Pass> createLLVMLoweringPass(
			ModelicaToLLVMConversionOptions options = ModelicaToLLVMConversionOptions::getDefaultOptions(),
			unsigned int bitWidth = 64);

	inline void registerLLVMLoweringPass()
	{
		mlir::registerPass("convert-to-llvm", "Modelica: LLVM lowering pass",
											 []() -> std::unique_ptr<::mlir::Pass> {
												 return createLLVMLoweringPass();
											 });
	}
}

#endif // MARCO_CODEN_CONVERSION_MODELICA_LOWERTOLLVM_H
