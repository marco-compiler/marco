#include "marco/Dialect/SUNDIALS/Attributes.h"
#include "marco/Dialect/SUNDIALS/SUNDIALSDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::sundials;

//===---------------------------------------------------------------------===//
// Tablegen attribute definitions
//===---------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/SUNDIALS/SUNDIALSAttributes.cpp.inc"
