#ifndef MARCO_DIALECT_MODELICA_TRANSFORMS_EXPANDCONNECTIONS_H
#define MARCO_DIALECT_MODELICA_TRANSFORMS_EXPANDCONNECTIONS_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica {
#define GEN_PASS_DECL_EXPANDCONNECTIONSPASS
#include "marco/Dialect/Modelica/Transforms/Passes.h.inc"
} // namespace mlir::modelica

#endif // MARCO_DIALECT_MODELICA_TRANSFORMS_EXPANDCONNECTIONS_H
