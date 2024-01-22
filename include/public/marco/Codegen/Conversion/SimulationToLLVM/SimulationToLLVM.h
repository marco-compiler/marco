#ifndef MARCO_CODEGEN_CONVERSION_SIMULATIONTOFUNC_SIMULATIONTOLLVM_H
#define MARCO_CODEGEN_CONVERSION_SIMULATIONTOFUNC_SIMULATIONTOFUNC_H

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DECL_SIMULATIONTOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createSimulationToLLVMConversionPass();

  std::unique_ptr<mlir::Pass> createSimulationToLLVMConversionPass(
      const SimulationToLLVMConversionPassOptions& options);
}

#endif // MARCO_CODEGEN_CONVERSION_SIMULATIONTOFUNC_SIMULATIONTOLLVM_H
