#pragma once

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
	/**
	 * Create a pass to convert Ida operations to a mix of Std,
	 * SCF and LLVM ones.
	 *
	 * @param options conversion options
	 * @param bitWidth bit width
 	 */
	std::unique_ptr<mlir::Pass> createIdaConversionPass(unsigned int bitWidth = 64);

	inline void registerIdaConversionPass()
	{
		mlir::registerPass("convert-ida", "Ida: conversion to std + scf + llvm dialects",
											 []() -> std::unique_ptr<::mlir::Pass> {
                          // TODO
                          return nullptr;
												 //return createIdaConversionPass();
											 });
	}
}
