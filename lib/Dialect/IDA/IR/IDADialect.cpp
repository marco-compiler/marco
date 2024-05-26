#include "marco/Dialect/IDA/IR/IDADialect.h"

using namespace ::mlir::ida;

#include "marco/Dialect/IDA/IR/IDADialect.cpp.inc"

//===---------------------------------------------------------------------===//
// IDA dialect
//===---------------------------------------------------------------------===//

namespace mlir::ida
{
  void IDADialect::initialize()
  {
    addOperations<
#define GET_OP_LIST
#include "marco/Dialect/IDA/IR/IDA.cpp.inc"
        >();

    addTypes<
#define GET_TYPEDEF_LIST
#include "marco/Dialect/IDA/IR/IDATypes.cpp.inc"
        >();
  }
}
