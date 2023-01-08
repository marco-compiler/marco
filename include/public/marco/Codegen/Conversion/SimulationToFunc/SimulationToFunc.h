#ifndef MARCO_CODEGEN_CONVERSION_SIMULATIONTOFUNC_SIMULATIONTOFUNC_H
#define MARCO_CODEGEN_CONVERSION_SIMULATIONTOFUNC_SIMULATIONTOFUNC_H

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DECL_SIMULATIONTOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  void populateSimulationToFuncStructuralTypeConversionsAndLegality(
      mlir::LLVMTypeConverter& typeConverter,
      mlir::RewritePatternSet& patterns,
      mlir::ConversionTarget& target);

  std::unique_ptr<mlir::Pass> createSimulationToFuncConversionPass();

  std::unique_ptr<mlir::Pass> createSimulationToFuncConversionPass(
      const SimulationToFuncConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_SIMULATIONTOFUNC_SIMULATIONTOFUNC_H
