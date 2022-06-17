#ifndef MARCO_CODEN_CONVERSION_MODELICA_MODELICACONVERSION_H
#define MARCO_CODEN_CONVERSION_MODELICA_MODELICACONVERSION_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
	struct ModelicaConversionOptions
	{
		bool assertions = true;
		bool outputArraysPromotion = true;

		static const ModelicaConversionOptions& getDefaultOptions() {
			static ModelicaConversionOptions options;
			return options;
		}
	};

	/// Create a pass to convert Modelica operations to a mix of Std,
	/// SCF and LLVM ones.
	std::unique_ptr<mlir::Pass> createModelicaConversionPass(
			ModelicaConversionOptions options = ModelicaConversionOptions::getDefaultOptions(),
			unsigned int bitWidth = 64);
}

#endif // MARCO_CODEN_CONVERSION_MODELICA_MODELICACONVERSION_H
