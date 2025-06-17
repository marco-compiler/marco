#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICATOMLIRCORE_BASEMODELICATOMLIRCORE_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICATOMLIRCORE_BASEMODELICATOMLIRCORE_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DECL_BASEMODELICATOMLIRCORECONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createBaseModelicaToMLIRCoreConversionPass();
} // namespace mlir

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICATOMLIRCORE_BASEMODELICATOMLIRCORE_H
