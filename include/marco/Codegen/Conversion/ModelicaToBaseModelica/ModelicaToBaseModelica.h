#ifndef MARCO_CODEGEN_CONVERSION_MODELICATOBASEMODELICA_MODELICATOBASEMODELICA_H
#define MARCO_CODEGEN_CONVERSION_MODELICATOBASEMODELICA_MODELICATOBASEMODELICA_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_MODELICATOBASEMODELICACONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createModelicaToBaseModelicaConversionPass();
} // namespace mlir

#endif // MARCO_CODEGEN_CONVERSION_MODELICATOBASEMODELICA_MODELICATOBASEMODELICA_H
