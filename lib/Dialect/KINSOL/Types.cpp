#include "marco/Dialect/KINSOL/Types.h"
#include "marco/Dialect/KINSOL/KINSOLDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::kinsol;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/KINSOL/KINSOLTypes.cpp.inc"
