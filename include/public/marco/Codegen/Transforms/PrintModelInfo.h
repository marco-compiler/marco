#ifndef MARCO_CODEGEN_TRANSFORMS_PRINTMODELINFO_H
#define MARCO_CODEGEN_TRANSFORMS_PRINTMODELINFO_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_PRINTMODELINFOPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createPrintModelInfoPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_PRINTMODELINFO_H
