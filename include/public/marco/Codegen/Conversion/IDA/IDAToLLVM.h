#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace marco::codegen
{
	std::unique_ptr<mlir::Pass> createIDAToLLVMConversionPass();

	inline void registerIDAToLLVMConversionPass()
	{
		mlir::registerPass(
        "ida-to-llvm", "IDA: conversion to Std + LLVM dialect",
        []() -> std::unique_ptr<::mlir::Pass> {
          return createIDAToLLVMConversionPass();
        });
	}

  void populateIDAStructuralTypeConversionsAndLegality(
      mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns, mlir::ConversionTarget& target);
}
