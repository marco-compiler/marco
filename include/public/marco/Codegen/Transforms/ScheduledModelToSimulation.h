#ifndef MARCO_CODEGEN_TRANSFORMS_SCHEDULEDMODELTOSIMULATION_H
#define MARCO_CODEGEN_TRANSFORMS_SCHEDULEDMODELTOSIMULATION_H

#include "marco/Codegen/Transforms/ModelSolving/ExternalSolvers/IDAOptions.h"
#include "marco/Codegen/Transforms/ModelSolving/Solver.h"
#include "marco/VariableFilter/VariableFilter.h"
#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_CONVERTSCHEDULEDMODELTOSIMULATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createConvertScheduledModelToSimulationPass();

  std::unique_ptr<mlir::Pass> createConvertScheduledModelToSimulationPass(
      const ConvertScheduledModelToSimulationPassOptions& options,
      marco::VariableFilter* variableFilter,
      marco::codegen::Solver solver,
      marco::codegen::IDAOptions idaOptions);
}

#endif // MARCO_CODEGEN_TRANSFORMS_SCHEDULEDMODELTOSIMULATION_H
