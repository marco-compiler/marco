#ifndef MARCO_CODEGEN_TRANSFORMS_MODELCONVERSION_H
#define MARCO_CODEGEN_TRANSFORMS_MODELCONVERSION_H

#include "marco/Codegen/Transforms/ModelSolving/Solver.h"
#include "marco/VariableFilter/VariableFilter.h"
#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_MODELCONVERSIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createModelConversionPass();

  std::unique_ptr<mlir::Pass> createModelConversionPass(const ModelConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELCONVERSION_H
