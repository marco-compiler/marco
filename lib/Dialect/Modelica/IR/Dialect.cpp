#include "marco/Dialect/Modelica/IR/Modelica.h"

using namespace ::mlir::modelica;

#include "marco/Dialect/Modelica/IR/Modelica.cpp.inc"

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
