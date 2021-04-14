#pragma once

#include <mlir/Pass/Pass.h>

namespace modelica::codegen
{
	struct ModelicaToLLVMConversionOptions
	{
		bool emitCWrappers = false;

		static const ModelicaToLLVMConversionOptions& getDefaultOptions() {
			static ModelicaToLLVMConversionOptions options;
			return options;
		}
	};

	std::unique_ptr<mlir::Pass> createLLVMLoweringPass(ModelicaToLLVMConversionOptions options = ModelicaToLLVMConversionOptions::getDefaultOptions());
}
