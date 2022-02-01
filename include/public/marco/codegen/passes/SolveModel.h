#pragma once

#include "marco/utils/VariableFilter.h"
#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
	enum Solver {
		ForwardEuler, CleverDAE
	};

	struct SolveModelOptions
	{
		bool emitMain = true;
		Solver solver = ForwardEuler;
    marco::VariableFilter* variableFilter = nullptr;

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
