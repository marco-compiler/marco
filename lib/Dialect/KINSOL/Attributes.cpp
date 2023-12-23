#include "marco/Dialect/KINSOL/Attributes.h"
#include "marco/Dialect/KINSOL/KINSOLDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::kinsol;

//===---------------------------------------------------------------------===//
// Tablegen attribute definitions
//===---------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/KINSOL/KINSOLAttributes.cpp.inc"
