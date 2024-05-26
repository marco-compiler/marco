#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONACCESSSPLIT_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONACCESSSPLIT_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_EQUATIONACCESSSPLITPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createEquationAccessSplitPass();
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONACCESSSPLIT_H
