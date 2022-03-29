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

	/**
	 * Create a pass to convert Modelica operations to a mix of Std,
	 * SCF and LLVM ones.
	 *
	 * @param options  conversion options
	 * @param bitWidth bit width
 	 */
	std::unique_ptr<mlir::Pass> createModelicaConversionPass(
			ModelicaConversionOptions options = ModelicaConversionOptions::getDefaultOptions(),
			unsigned int bitWidth = 64);

	inline void registerModelicaConversionPass()
	{
		mlir::registerPass("convert-modelica", "Modelica: conversion to std + scf + llvm dialects",
											 []() -> std::unique_ptr<::mlir::Pass> {
												 return createModelicaConversionPass();
											 });
	}
}

#endif // MARCO_CODEN_CONVERSION_MODELICA_MODELICACONVERSION_H
