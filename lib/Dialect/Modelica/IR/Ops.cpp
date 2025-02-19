#include "marco/Dialect/Modelica/IR/Ops.h"
#include "marco/Dialect/Modelica/IR/Modelica.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace ::mlir::modelica;

//===---------------------------------------------------------------------===//
// Modelica Dialect
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
void ModelicaDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "marco/Dialect/Modelica/IR/ModelicaOps.cpp.inc"
      >();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Modelica Operations
//===---------------------------------------------------------------------===//
