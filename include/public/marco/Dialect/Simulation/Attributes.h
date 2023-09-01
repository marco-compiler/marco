#ifndef MARCO_DIALECTS_SIMULATION_ATTRIBUTES_H
#define MARCO_DIALECTS_SIMULATION_ATTRIBUTES_H

#include "marco/Dialect/Modeling/Attributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Hashing.h"

namespace mlir::simulation
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
#include "marco/Dialect/Simulation/SimulationAttributes.h.inc"

#endif // MARCO_DIALECTS_SIMULATION_ATTRIBUTES_H
