#include "marco/Dialect/Runtime/IR/Runtime.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::runtime;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/Runtime/IR/RuntimeTypes.cpp.inc"

//===---------------------------------------------------------------------===//
// RuntimeDialect
//===---------------------------------------------------------------------===//

namespace mlir::runtime {
void RuntimeDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "marco/Dialect/Runtime/IR/RuntimeTypes.cpp.inc"

      >();
}
} // namespace mlir::runtime
