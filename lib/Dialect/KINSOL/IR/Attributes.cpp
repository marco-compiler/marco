#include "marco/Dialect/KINSOL/IR/Attributes.h"
#include "marco/Dialect/KINSOL/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::kinsol;

//===---------------------------------------------------------------------===//
// Tablegen attribute definitions
//===---------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/KINSOL/IR/Attributes.cpp.inc"
