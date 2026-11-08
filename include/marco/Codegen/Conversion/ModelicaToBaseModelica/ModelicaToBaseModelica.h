#ifndef MARCO_CODEGEN_CONVERSION_MODELICATOBASEMODELICA_MODELICATOBASEMODELICA_H
#define MARCO_CODEGEN_CONVERSION_MODELICATOBASEMODELICA_MODELICATOBASEMODELICA_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_MODELICATOBASEMODELICACONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

/// Convert the flat models and the functions of the Modelica dialect
/// contained in the module into Base Modelica operations. The converted
/// Modelica operations are erased.
mlir::LogicalResult convertModelicaToBaseModelica(mlir::ModuleOp moduleOp);

std::unique_ptr<mlir::Pass> createModelicaToBaseModelicaConversionPass();
} // namespace mlir

#endif // MARCO_CODEGEN_CONVERSION_MODELICATOBASEMODELICA_MODELICATOBASEMODELICA_H
