// #include "marco/Dialect/Runtime/IR/Types.h"
#include "marco/Dialect/Runtime/IR/Runtime.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinDialect.h"
// #include "mlir/IR/DialectImplementation.h"
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

namespace mlir::runtime
{
  void RuntimeDialect::registerTypes()
  {
    addTypes<
      #define GET_TYPEDEF_LIST
      #include "marco/Dialect/Runtime/IR/RuntimeTypes.cpp.inc"
    >();
  }
} // namespace mlir::runtime
