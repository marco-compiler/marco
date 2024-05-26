#include "marco/Dialect/SUNDIALS/IR/SUNDIALSDialect.h"

using namespace ::mlir::sundials;

#include "marco/Dialect/SUNDIALS/IR/SUNDIALSDialect.cpp.inc"

//===---------------------------------------------------------------------===//
// SUNDIALS dialect
//===---------------------------------------------------------------------===//

namespace mlir::sundials
{
  void SUNDIALSDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "marco/Dialect/SUNDIALS/IR/SUNDIALS.cpp.inc"
        >();

    addTypes<
#define GET_TYPEDEF_LIST
#include "marco/Dialect/SUNDIALS/IR/SUNDIALSTypes.cpp.inc"
        >();
  }
}
