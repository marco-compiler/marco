#include "marco/Dialect/Modelica/IR/Attributes.h"
#include "marco/Dialect/Modelica/IR/Modelica.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::modelica;

//===----------------------------------------------------------------------===//
// Tablegen attribute definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/Modelica/IR/ModelicaAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

namespace mlir::modelica {
void ModelicaDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "marco/Dialect/Modelica/IR/ModelicaAttributes.cpp.inc"

      >();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Attributes
//===---------------------------------------------------------------------===//
