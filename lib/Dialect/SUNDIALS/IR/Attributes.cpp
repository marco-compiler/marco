#include "marco/Dialect/SUNDIALS/IR/Attributes.h"
#include "marco/Dialect/SUNDIALS/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::sundials;

//===---------------------------------------------------------------------===//
// Tablegen attribute definitions
//===---------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/SUNDIALS/IR/Attributes.cpp.inc"
