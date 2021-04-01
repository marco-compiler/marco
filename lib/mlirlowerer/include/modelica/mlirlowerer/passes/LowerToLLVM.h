#pragma once

#include <mlir/Pass/Pass.h>

namespace modelica::codegen
{
	struct ModelicaToLLVMConversionOptions
	{
		bool emitCWrappers = false;

		/**
		 * Get a statically allocated copy of the default options.
		 *
		 * @return default options
		 */
		static const ModelicaToLLVMConversionOptions& getDefaultOptions() {
			static ModelicaToLLVMConversionOptions options;
			return options;
		}
	};

	std::unique_ptr<mlir::Pass> createLLVMLoweringPass(ModelicaToLLVMConversionOptions options = ModelicaToLLVMConversionOptions::getDefaultOptions());
}
