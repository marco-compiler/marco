#ifndef MARCO_DIALECT_MODELICA_COMMON_H
#define MARCO_DIALECT_MODELICA_COMMON_H

#include "marco/Dialect/Modeling/ModelingDialect.h"

namespace mlir::modelica
{
  using Point = ::mlir::modeling::Point;
  using Range = ::mlir::modeling::Range;
  using MultidimensionalRange = ::mlir::modeling::MultidimensionalRange;
  using IndexSet = ::mlir::modeling::IndexSet;
}

#endif // MARCO_DIALECT_MODELICA_COMMON_H
