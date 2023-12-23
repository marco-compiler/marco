#include "marco/Dialect/IDA/Types.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::ida;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/IDA/IDATypes.cpp.inc"
