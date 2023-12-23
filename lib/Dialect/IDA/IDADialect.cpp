#include "marco/Dialect/IDA/IDADialect.h"

using namespace ::mlir::ida;

#include "marco/Dialect/IDA/IDADialect.cpp.inc"

//===---------------------------------------------------------------------===//
// IDA dialect
//===---------------------------------------------------------------------===//

namespace mlir::ida
{
  void IDADialect::initialize()
  {
    addOperations<
#define GET_OP_LIST
#include "marco/Dialect/IDA/IDA.cpp.inc"
        >();

    addTypes<
#define GET_TYPEDEF_LIST
#include "marco/Dialect/IDA/IDATypes.cpp.inc"
        >();
  }
}
