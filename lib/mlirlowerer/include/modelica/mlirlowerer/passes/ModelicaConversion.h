#pragma once

#include <mlir/Pass/Pass.h>

namespace modelica::codegen
{
	struct ModelicaConversionOptions
	{
		bool assertions = true;
		bool useRuntimeLibrary = true;

		static const ModelicaConversionOptions& getDefaultOptions() {
			static ModelicaConversionOptions options;
			return options;
		}
	};

	/**
	 * Convert the Modelica functions into functions of the built-in dialect.
	 */
	std::unique_ptr<mlir::Pass> createFunctionConversionPass();

	inline void registerFunctionConversionPass()
	{
		mlir::registerPass("convert-modelica-functions", "Modelica: functions lowering",
											 []() -> std::unique_ptr<::mlir::Pass> {
												 return createFunctionConversionPass();
											 });
	}

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
