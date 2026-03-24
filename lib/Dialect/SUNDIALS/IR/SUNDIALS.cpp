#include "marco/Dialect/SUNDIALS/IR/Dialect.h"

using namespace ::mlir::sundials;

#include "marco/Dialect/SUNDIALS/IR/Dialect.cpp.inc"

//===---------------------------------------------------------------------===//
// SUNDIALS dialect
//===---------------------------------------------------------------------===//

namespace mlir::sundials {
void SUNDIALSDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "marco/Dialect/SUNDIALS/IR/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "marco/Dialect/SUNDIALS/IR/Types.cpp.inc"
      >();
}
} // namespace mlir::sundials
