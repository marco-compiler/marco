#include "marco/Dialect/IDA/IR/Types.h"
#include "marco/Dialect/IDA/IR/IDADialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::ida;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/IDA/IR/IDATypes.cpp.inc"
