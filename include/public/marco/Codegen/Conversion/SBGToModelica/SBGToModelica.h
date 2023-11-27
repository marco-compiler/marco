#ifndef MARCO_CODEGEN_CONVERSION_SBGTOMODELICA_SBGTOMODELICA_H
#define MARCO_CODEGEN_CONVERSION_SBGTOMODELICA_SBGTOMODELICA_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/SBG/SBGDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DECL_SBGTOMODELICACONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createSBGToModelicaConversionPass();
}

#endif // MARCO_CODEGEN_CONVERSION_SBGTOMODELICA_SBGTOMODELICA_H
