#include "marco/Dialect/Modeling/Types.h"
#include "marco/Dialect/Modeling/ModelingDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::modeling;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/Modeling/ModelingTypes.cpp.inc"
