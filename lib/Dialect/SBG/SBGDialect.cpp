#include "marco/Dialect/SBG/SBGDialect.h"
#include "marco/Dialect/SBG/Ops.h"
#include "marco/Dialect/SBG/Attributes.h"
#include "marco/Dialect/SBG/Types.h"
#include "mlir/IR/DialectImplementation.h"
#include "sbg/map.hpp"

using namespace ::mlir;
using namespace ::mlir::sbg;

#include "marco/Dialect/SBG/SBGDialect.cpp.inc"

//===---------------------------------------------------------------------===//
// SBG dialect
//===---------------------------------------------------------------------===//

namespace mlir::sbg
{
  void SBGDialect::initialize()
  {
    registerTypes();
    registerAttributes();

    addOperations<
      #define GET_OP_LIST
      #include "marco/Dialect/SBG/SBG.cpp.inc"
    >();
  }
}
