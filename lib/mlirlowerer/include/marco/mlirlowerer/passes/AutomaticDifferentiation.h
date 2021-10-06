#pragma once

#include <marco/mlirlowerer/passes/SolveModel.h>
#include <mlir/Pass/Pass.h>

namespace marco::codegen
{
	std::unique_ptr<mlir::Pass> createAutomaticDifferentiationPass(
			SolveModelOptions options = SolveModelOptions::getDefaultOptions());

	inline void registerAutomaticDifferentiationPass()
	{
		mlir::registerPass("auto-diff", "Modelica: automatic differentiation of functions",
											 []() -> std::unique_ptr<::mlir::Pass> {
												 return createAutomaticDifferentiationPass();
											 });
	}
}
