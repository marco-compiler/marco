#include "marco/Dialect/Modelica/IR/Types.h"
#include "marco/Dialect/Modelica/IR/Modelica.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::modelica;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/Modelica/IR/ModelicaTypes.cpp.inc"

//===---------------------------------------------------------------------===//
// Dialect
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
void ModelicaDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "marco/Dialect/Modelica/IR/ModelicaTypes.cpp.inc"

      >();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Types
//===---------------------------------------------------------------------===//
