#ifndef MARCO_DIALECT_KINSOL_IR_ATTRIBUTES_H
#define MARCO_DIALECT_KINSOL_IR_ATTRIBUTES_H

#include "marco/Dialect/Modeling/IR/Attributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"

namespace mlir::kinsol {
using RangeAttr = ::mlir::modeling::RangeAttr;
using MultidimensionalRangeAttr = ::mlir::modeling::MultidimensionalRangeAttr;
} // namespace mlir::kinsol

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/KINSOL/IR/KINSOLAttributes.h.inc"

#endif // MARCO_DIALECT_KINSOL_IR_ATTRIBUTES_H
