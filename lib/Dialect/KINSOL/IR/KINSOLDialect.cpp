#include "marco/Dialect/KINSOL/IR/KINSOLDialect.h"

using namespace ::mlir::kinsol;

#include "marco/Dialect/KINSOL/IR/KINSOLDialect.cpp.inc"

//===---------------------------------------------------------------------===//
// KINSOL dialect
//===---------------------------------------------------------------------===//

namespace mlir::kinsol
{
  void KINSOLDialect::initialize()
  {
    addOperations<
#define GET_OP_LIST
#include "marco/Dialect/KINSOL/IR/KINSOL.cpp.inc"
        >();

    addTypes<
#define GET_TYPEDEF_LIST
#include "marco/Dialect/KINSOL/IR/KINSOLTypes.cpp.inc"
        >();
  }
}
