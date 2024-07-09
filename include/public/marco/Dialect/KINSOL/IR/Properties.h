#ifndef MARCO_DIALECT_KINSOL_IR_PROPERTIES_H
#define MARCO_DIALECT_KINSOL_IR_PROPERTIES_H

#include "marco/Dialect/Modeling/IR/Properties.h"

namespace mlir::kinsol
{
  using Point = ::mlir::modeling::Point;
  using Range = ::mlir::modeling::Range;
  using MultidimensionalRange = ::mlir::modeling::MultidimensionalRange;
  using IndexSet = ::mlir::modeling::IndexSet;
}

#endif // MARCO_DIALECT_KINSOL_IR_PROPERTIES_H
