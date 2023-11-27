#include "marco/Dialect/SBG/SBGDialect.h"
#include "marco/Dialect/SBG/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::sbg;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/SBG/SBGTypes.cpp.inc"

namespace mlir::sbg
{
  void SBGDialect::registerTypes()
  {
    addTypes<
      #define GET_TYPEDEF_LIST
      #include "marco/Dialect/SBG/SBGTypes.cpp.inc"
    >();
  }
}