#ifndef MARCO_CODEGEN_TRANSFORMS_SIMULATIONVARIABLESINSERTION_H
#define MARCO_CODEGEN_TRANSFORMS_SIMULATIONVARIABLESINSERTION_H

#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_SSIMULATIONVARIABLESINSERTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createSimulationVariablesInsertionPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_SIMULATIONVARIABLESINSERTION_H
