#pragma once

#include <mlir/Pass/Pass.h>

namespace modelica::codegen
{
	struct ModelicaConversionOptions
	{
		bool useRuntimeLibrary = true;

		static const ModelicaConversionOptions& getDefaultOptions() {
			static ModelicaConversionOptions options;
			return options;
		}
	};

	/**
	 * Create a pass to convert Modelica operations to a mix of Std,
	 * SCF and LLVM ones.
	 *
	 * @param options  conversion options
 	 */
	std::unique_ptr<mlir::Pass> createModelicaConversionPass(ModelicaConversionOptions options = ModelicaConversionOptions::getDefaultOptions(), unsigned int bitWidth = 64);

	/**
	 * Convert the control flow operations of the Modelica and the SCF
	 * dialects.
	 *
	 * @param options  conversion options
	 */
	std::unique_ptr<mlir::Pass> createLowerToCFGPass(ModelicaConversionOptions options = ModelicaConversionOptions::getDefaultOptions());
}
