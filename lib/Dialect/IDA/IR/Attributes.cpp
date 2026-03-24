#include "marco/Dialect/IDA/IR/Attributes.h"
#include "marco/Dialect/IDA/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::ida;

//===---------------------------------------------------------------------===//
// Tablegen attribute definitions
//===---------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/IDA/IR/Attributes.cpp.inc"
