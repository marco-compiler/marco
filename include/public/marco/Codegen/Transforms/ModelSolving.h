#ifndef MARCO_CODEN_TRANSFORMS_MODELSOLVING_H
#define MARCO_CODEN_TRANSFORMS_MODELSOLVING_H

#include "marco/Codegen/Transforms/ModelSolving/ExternalSolvers/IDAOptions.h"
#include "marco/VariableFilter/VariableFilter.h"
#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
  enum class Solver
  {
    forwardEuler,
    ida
  };

  struct ModelSolvingOptions
  {
    double startTime = 0;
    double endTime = 10;
    double timeStep = 0.1;

    bool emitMain = true;
    marco::VariableFilter* variableFilter = nullptr;

    Solver solver = Solver::forwardEuler;
    IDAOptions ida;

    static const ModelSolvingOptions& getDefaultOptions() {
      static ModelSolvingOptions options;
      return options;
    }
  };

	std::unique_ptr<mlir::Pass> createModelSolvingPass(
      ModelSolvingOptions options = ModelSolvingOptions::getDefaultOptions(),
			unsigned int bitWidth = 64);
}

#endif // MARCO_CODEN_TRANSFORMS_MODELSOLVING_H
