#include "marco/Dialect/IDA/IR/Dialect.h"

using namespace ::mlir::ida;

#include "marco/Dialect/IDA/IR/Dialect.cpp.inc"

//===---------------------------------------------------------------------===//
// IDA dialect
//===---------------------------------------------------------------------===//

namespace mlir::ida {
void IDADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "marco/Dialect/IDA/IR/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "marco/Dialect/IDA/IR/Types.cpp.inc"
      >();
}
} // namespace mlir::ida
