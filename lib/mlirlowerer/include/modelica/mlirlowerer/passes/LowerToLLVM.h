#pragma once

#include <mlir/Pass/Pass.h>

namespace modelica::codegen
{
	struct ModelicaToLLVMConversionOptions
	{
		bool assertions = true;
		bool emitCWrappers = false;

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
