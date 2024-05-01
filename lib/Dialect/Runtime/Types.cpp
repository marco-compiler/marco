#include "marco/Dialect/Runtime/Types.h"
#include "marco/Dialect/Runtime/RuntimeDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::runtime;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/Runtime/RuntimeTypes.cpp.inc"
