#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_RECORDINLINING_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_RECORDINLINING_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_RECORDINLININGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createRecordInliningPass();
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_RECORDINLINING_H
