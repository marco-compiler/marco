#pragma once

#include <mlir/Pass/Pass.h>

namespace modelica::codegen
{
	struct FunctionsScalarizationOptions
	{
		bool assertions = true;

		static const FunctionsScalarizationOptions& getDefaultOptions() {
			static FunctionsScalarizationOptions options;
			return options;
		}
	};

	std::unique_ptr<mlir::Pass> createFunctionsScalarizationPass(
			FunctionsScalarizationOptions options = FunctionsScalarizationOptions::getDefaultOptions());

	inline void registerFunctionsScalarizationPass()
	{
		mlir::registerPass("scalarize-functions", "Convert vectorized functions in loops with scalar calls",
											 []() -> std::unique_ptr<::mlir::Pass> {
												 return createFunctionsScalarizationPass();
											 });
	}
}
