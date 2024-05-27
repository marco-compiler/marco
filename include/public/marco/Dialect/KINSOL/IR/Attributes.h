#ifndef MARCO_DIALECT_KINSOL_IR_ATTRIBUTES_H
#define MARCO_DIALECT_KINSOL_IR_ATTRIBUTES_H

#include "marco/Dialect/Modeling/IR/Attributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/StorageUniquer.h"

namespace mlir::kinsol
{
  using Point = ::mlir::modeling::Point;
  using Range = ::mlir::modeling::Range;
  using MultidimensionalRange = ::mlir::modeling::MultidimensionalRange;
  using IndexSet = ::mlir::modeling::IndexSet;

  using RangeAttr = ::mlir::modeling::RangeAttr;
  using MultidimensionalRangeAttr = ::mlir::modeling::MultidimensionalRangeAttr;
  using IndexSetAttr = ::mlir::modeling::IndexSetAttr;
}

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/KINSOL/IR/KINSOLAttributes.h.inc"

#endif // MARCO_DIALECT_KINSOL_IR_ATTRIBUTES_H
