#ifndef MARCO_CODEGEN_TRANSFORMS_RECORDINLINING_H
#define MARCO_CODEGEN_TRANSFORMS_RECORDINLINING_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_RECORDINLININGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createRecordInliningPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_RECORDINLINING_H
