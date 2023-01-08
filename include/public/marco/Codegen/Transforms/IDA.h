#ifndef MARCO_CODEGEN_TRANSFORMS_IDA_H
#define MARCO_CODEGEN_TRANSFORMS_IDA_H

#include "marco/VariableFilter/VariableFilter.h"
#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_IDAPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createIDAPass();

  std::unique_ptr<mlir::Pass> createIDAPass(const IDAPassOptions& options);
}

#endif // MARCO_CODEGEN_TRANSFORMS_IDA_H
