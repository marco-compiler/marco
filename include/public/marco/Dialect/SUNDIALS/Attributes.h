#ifndef MARCO_DIALECTS_SUNDIALS_ATTRIBUTES_H
#define MARCO_DIALECTS_SUNDIALS_ATTRIBUTES_H

#include "marco/Dialect/Modeling/Attributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/StorageUniquer.h"

namespace mlir::sundials
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
#include "marco/Dialect/SUNDIALS/SUNDIALSAttributes.h.inc"

#endif // MARCO_DIALECTS_SUNDIALS_ATTRIBUTES_H
