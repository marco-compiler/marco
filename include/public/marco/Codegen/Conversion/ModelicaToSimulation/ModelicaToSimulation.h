#ifndef MARCO_CODEGEN_CONVERSION_MODELICATOSIMULATION_MODELICATOSIMULATION_H
#define MARCO_CODEGEN_CONVERSION_MODELICATOSIMULATION_MODELICATOSIMULATION_H

#include "mlir/Pass/Pass.h"

namespace mlir
{
#define GEN_PASS_DECL_MODELICATOSIMULATIONCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createModelicaToSimulationConversionPass();

  std::unique_ptr<mlir::Pass> createModelicaToSimulationConversionPass(
      const ModelicaToSimulationConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICATOSIMULATION_MODELICATOSIMULATION_H
