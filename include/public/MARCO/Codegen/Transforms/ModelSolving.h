#ifndef MARCO_CODEN_TRANSFORMS_MODELSOLVING_H
#define MARCO_CODEN_TRANSFORMS_MODELSOLVING_H

#include "marco/Utils/VariableFilter.h"
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
		mlir::registerPass(
        "solve-model", "Modelica: solve model",
        []() -> std::unique_ptr<::mlir::Pass> {
          return createSolveModelPass();
        });
	}
}

#endif // MARCO_CODEN_TRANSFORMS_MODELSOLVING_H
