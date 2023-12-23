#include "marco/Dialect/SUNDIALS/Types.h"
#include "marco/Dialect/SUNDIALS/SUNDIALSDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::sundials;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/SUNDIALS/SUNDIALSTypes.cpp.inc"
