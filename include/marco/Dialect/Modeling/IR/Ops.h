#ifndef MARCO_DIALECT_MODELING_IR_OPS_H
#define MARCO_DIALECT_MODELING_IR_OPS_H

#include "marco/Dialect/Modeling/IR/Attributes.h"
#include "marco/Dialect/Modeling/IR/Properties.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "marco/Dialect/Modeling/IR/ModelingOps.h.inc"

namespace mlir::modeling {
bool parseIndexSet(mlir::OpAsmParser &parser, IndexSet &prop);

void printIndexSet(mlir::OpAsmPrinter &printer, mlir::Operation *op,
                   const IndexSet &prop);
} // namespace mlir::modeling

#endif // MARCO_DIALECT_MODELING_IR_OPS_H
