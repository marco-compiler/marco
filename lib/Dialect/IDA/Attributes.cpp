#include "marco/Dialect/IDA/Attributes.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::ida;

//===---------------------------------------------------------------------===//
// Tablegen attribute definitions
//===---------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/IDA/IDAAttributes.cpp.inc"
