#ifndef MARCO_DIALECT_IDA_IR_ATTRIBUTES_H
#define MARCO_DIALECT_IDA_IR_ATTRIBUTES_H

#include "marco/Dialect/Modeling/IR/Attributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/StorageUniquer.h"

namespace mlir::ida
{
  using RangeAttr = ::mlir::modeling::RangeAttr;
  using MultidimensionalRangeAttr = ::mlir::modeling::MultidimensionalRangeAttr;
}

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/IDA/IR/IDAAttributes.h.inc"

#endif // MARCO_DIALECT_IDA_IR_ATTRIBUTES_H
