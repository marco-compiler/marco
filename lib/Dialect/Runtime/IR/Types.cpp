#include "marco/Dialect/Runtime/IR/Types.h"
#include "marco/Dialect/Runtime/IR/RuntimeDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::runtime;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/Runtime/IR/RuntimeTypes.cpp.inc"
