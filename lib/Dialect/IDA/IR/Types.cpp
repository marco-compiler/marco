#include "marco/Dialect/IDA/IR/Types.h"
#include "marco/Dialect/IDA/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::ida;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/IDA/IR/Types.cpp.inc"
