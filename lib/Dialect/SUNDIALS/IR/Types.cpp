#include "marco/Dialect/SUNDIALS/IR/Types.h"
#include "marco/Dialect/SUNDIALS/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::sundials;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/SUNDIALS/IR/Types.cpp.inc"
