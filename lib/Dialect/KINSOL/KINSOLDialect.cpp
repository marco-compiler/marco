#include "marco/Dialect/KINSOL/KINSOLDialect.h"

using namespace ::mlir::kinsol;

#include "marco/Dialect/KINSOL/KINSOLDialect.cpp.inc"

//===---------------------------------------------------------------------===//
// KINSOL dialect
//===---------------------------------------------------------------------===//

namespace mlir::kinsol
{
  void KINSOLDialect::initialize()
  {
    addOperations<
#define GET_OP_LIST
#include "marco/Dialect/KINSOL/KINSOL.cpp.inc"
        >();

    addTypes<
#define GET_TYPEDEF_LIST
#include "marco/Dialect/KINSOL/KINSOLTypes.cpp.inc"
        >();
  }
}
