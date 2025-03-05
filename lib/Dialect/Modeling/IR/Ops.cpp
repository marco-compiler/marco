#include "marco/Dialect/Modeling/IR/Ops.h"
#include "marco/Dialect/Modeling/IR/Modeling.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace ::mlir::modeling;

#define GET_OP_CLASSES
#include "marco/Dialect/Modeling/IR/ModelingOps.cpp.inc"

namespace mlir::modeling {
bool parseIndexSet(mlir::OpAsmParser &parser, IndexSet &prop) {
  return mlir::failed(parse(parser, prop));
}

void printIndexSet(mlir::OpAsmPrinter &printer, mlir::Operation *op,
                   const IndexSet &prop) {
  print(printer, prop);
}
} // namespace mlir::modeling
