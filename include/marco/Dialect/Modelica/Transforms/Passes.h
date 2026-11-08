#ifndef MARCO_DIALECT_MODELICA_TRANSFORMS_PASSES_H
#define MARCO_DIALECT_MODELICA_TRANSFORMS_PASSES_H

#include "marco/Dialect/Modelica/Transforms/ExpandConnections.h"
#include "marco/Dialect/Modelica/Transforms/Flatten.h"
#include "marco/Dialect/Modelica/Transforms/Instantiate.h"

namespace mlir::modelica {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "marco/Dialect/Modelica/Transforms/Passes.h.inc"
} // namespace mlir::modelica

#endif // MARCO_DIALECT_MODELICA_TRANSFORMS_PASSES_H
