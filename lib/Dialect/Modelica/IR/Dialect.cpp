#include "marco/Dialect/Modelica/IR/Dialect.h"

using namespace ::mlir::modelica;

#include "marco/Dialect/Modelica/IR/Dialect.cpp.inc"

namespace mlir::modelica {
//===-------------------------------------------------------------------===//
// Modelica dialect
//===-------------------------------------------------------------------===//

void ModelicaDialect::initialize() {
  registerTypes();
  registerAttributes();
  registerOperations();
}
} // namespace mlir::modelica
