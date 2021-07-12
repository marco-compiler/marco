#pragma once

#include <mlir/Pass/Pass.h>

namespace modelica::codegen
{
	enum Solver {
		ForwardEuler, CleverDAE
	};

	struct SolveModelOptions
	{
		bool emitMain = true;
		int matchingMaxIterations = 1000;
		int sccMaxIterations = 1000;
		Solver solver = ForwardEuler;

		static const SolveModelOptions& getDefaultOptions() {
			static SolveModelOptions options;
			return options;
		}
	};

	std::unique_ptr<mlir::Pass> createSolveModelPass(
			SolveModelOptions options = SolveModelOptions::getDefaultOptions(),
			unsigned int bitWidth = 64);

	inline void registerSolveModelPass()
	{
		mlir::registerPass("solve-model", "Modelica: solve model",
											 []() -> std::unique_ptr<::mlir::Pass> {
												 return createSolveModelPass();
											 });
	}
}
