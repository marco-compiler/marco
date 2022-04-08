#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace marco::codegen
{
	/// Create a pass to convert Ida operations to a mix of Std,
	/// SCF and LLVM ones.
	std::unique_ptr<mlir::Pass> createIDAConversionPass();

	inline void registerIDAConversionPass()
	{
		mlir::registerPass(
        "convert-ida", "IDA: conversion to Std + LLVM dialect",
        []() -> std::unique_ptr<::mlir::Pass> {
          return createIDAConversionPass();
        });
	}

  void populateIDAStructuralTypeConversionsAndLegality(
      mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns, mlir::ConversionTarget& target);
}
