#include "marco/Dialect/Modeling/IR/Types.h"
#include "marco/Dialect/Modeling/IR/ModelingDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::modeling;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/Modeling/IR/ModelingTypes.cpp.inc"
