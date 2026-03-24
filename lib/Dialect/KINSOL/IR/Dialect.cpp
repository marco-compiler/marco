#include "marco/Dialect/KINSOL/IR/Dialect.h"

using namespace ::mlir::kinsol;

#include "marco/Dialect/KINSOL/IR/Dialect.cpp.inc"

//===---------------------------------------------------------------------===//
// KINSOL dialect
//===---------------------------------------------------------------------===//

namespace mlir::kinsol {
void KINSOLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "marco/Dialect/KINSOL/IR/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "marco/Dialect/KINSOL/IR/Types.cpp.inc"
      >();
}
} // namespace mlir::kinsol
