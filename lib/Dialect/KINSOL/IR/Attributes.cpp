#include "marco/Dialect/KINSOL/IR/Attributes.h"
#include "marco/Dialect/KINSOL/IR/KINSOL.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::kinsol;

//===---------------------------------------------------------------------===//
// Tablegen attribute definitions
//===---------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/KINSOL/IR/KINSOLAttributes.cpp.inc"
