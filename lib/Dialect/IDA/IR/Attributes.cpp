#include "marco/Dialect/IDA/IR/Attributes.h"
#include "marco/Dialect/IDA/IR/IDA.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::ida;

//===---------------------------------------------------------------------===//
// Tablegen attribute definitions
//===---------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/IDA/IR/IDAAttributes.cpp.inc"
