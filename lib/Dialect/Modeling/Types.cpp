#include "marco/Dialect/Modeling/ModelingDialect.h"
#include "marco/Dialect/Modeling/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::modeling;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/Modeling/ModelingTypes.cpp.inc"
