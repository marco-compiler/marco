#include "marco/Dialect/Runtime/IR/Attributes.h"
#include "marco/Dialect/Runtime/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::runtime;

//===---------------------------------------------------------------------===//
// Tablegen attribute definitions
//===---------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/Runtime/IR/Attributes.cpp.inc"

//===---------------------------------------------------------------------===//
// RuntimeDialect
//===---------------------------------------------------------------------===//

namespace mlir::runtime {
void RuntimeDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "marco/Dialect/Runtime/IR/Attributes.cpp.inc"
      >();
}
} // namespace mlir::runtime

//===---------------------------------------------------------------------===//
// Attributes
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// VariableAttr

namespace mlir::runtime {
int64_t VariableAttr::getRank() const { return getDimensions().size(); }
} // namespace mlir::runtime
