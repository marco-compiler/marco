#pragma once

#include <marco/utils/VariableFilter.h>
#include <mlir/Pass/Pass.h>
#include <marco/mlirlowerer/passes/model/Model.h>

namespace marco::codegen
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
        marco::VariableFilter *variableFilter; // Variable Filter is used in solve model pass to filter out variables to be printed
		bool equidistantTimeGrid = false;
		bool printStatistics = false;

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

	/**
	 * This method must be used for testing purposes only.
	 * Given a parsed ModuleOp, it return the Model before the matching phase.
	 */
	llvm::Optional<model::Model> getUnmatchedModel(mlir::ModuleOp moduleOp, SolveModelOptions options);

	/**
	 * This method must be used for testing and debugging purposes only.
	 * Given a parsed ModuleOp, it return the Model at the end of the solving pass.
	 */
	llvm::Optional<model::Model> getSolvedModel(mlir::ModuleOp moduleOp, SolveModelOptions options);
}
