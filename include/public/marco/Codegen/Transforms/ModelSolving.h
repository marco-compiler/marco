#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_H

#include "marco/Codegen/Transforms/ModelSolving/ExternalSolvers/IDAOptions.h"
#include "marco/VariableFilter/VariableFilter.h"
#include "mlir/Pass/Pass.h"

namespace marco
{
  class VariableFilter;

  namespace codegen
  {
    enum class Solver
    {
      forwardEuler,
      ida
    };
  }
}

namespace mlir::modelica
{
#define GEN_PASS_DECL_MODELSOLVINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createModelSolvingPass();

  std::unique_ptr<mlir::Pass> createModelSolvingPass(
      const ModelSolvingPassOptions& options,
      marco::VariableFilter* variableFilter,
      marco::codegen::Solver solver,
      marco::codegen::IDAOptions idaOptions);
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_H
