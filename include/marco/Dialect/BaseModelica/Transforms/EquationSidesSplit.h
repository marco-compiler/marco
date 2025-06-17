#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONSIDESSPLIT_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONSIDESSPLIT_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_EQUATIONSIDESSPLITPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createEquationSidesSplitPass();
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONSIDESSPLIT_H
