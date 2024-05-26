#ifndef MARCO_DIALECT_BASEMODELICA_IR_COMMON_H
#define MARCO_DIALECT_BASEMODELICA_IR_COMMON_H

#include "marco/Dialect/Modeling/IR/ModelingDialect.h"

namespace mlir::bmodelica
{
  using Point = ::mlir::modeling::Point;
  using Range = ::mlir::modeling::Range;
  using MultidimensionalRange = ::mlir::modeling::MultidimensionalRange;
  using IndexSet = ::mlir::modeling::IndexSet;
}

#endif // MARCO_DIALECT_BASEMODELICA_IR_COMMON_H
