#include "marco/Dialect/KINSOL/IR/Types.h"
#include "marco/Dialect/KINSOL/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::kinsol;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/KINSOL/IR/Types.cpp.inc"
