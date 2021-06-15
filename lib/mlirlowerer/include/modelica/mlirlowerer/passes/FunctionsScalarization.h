#pragma once

#include <mlir/Pass/Pass.h>

namespace modelica::codegen
{
	std::unique_ptr<mlir::Pass> createFunctionsScalarizationPass();

	inline void registerFunctionsScalarizationPass()
	{
		mlir::registerPass("scalarize-functions", "Convert vectorized functions in loops with scalar calls",
											 []() -> std::unique_ptr<::mlir::Pass> {
												 return createFunctionsScalarizationPass();
											 });
	}
}
