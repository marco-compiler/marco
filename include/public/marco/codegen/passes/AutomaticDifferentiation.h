#pragma once

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
	std::unique_ptr<mlir::Pass> createAutomaticDifferentiationPass();

	inline void registerAutomaticDifferentiationPass()
	{
		mlir::registerPass("auto-diff", "Modelica: automatic differentiation of functions",
											 []() -> std::unique_ptr<::mlir::Pass> {
												 return createAutomaticDifferentiationPass();
											 });
	}
}
