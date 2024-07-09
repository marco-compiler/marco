#ifndef MARCO_DIALECT_RUNTIME_ATTRIBUTES_H
#define MARCO_DIALECT_RUNTIME_ATTRIBUTES_H

#include "marco/Dialect/Modeling/IR/Attributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Hashing.h"

namespace mlir::runtime
{
  using RangeAttr = ::mlir::modeling::RangeAttr;
  using MultidimensionalRangeAttr = ::mlir::modeling::MultidimensionalRangeAttr;
}

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/Runtime/IR/RuntimeAttributes.h.inc"

#endif // MARCO_DIALECT_RUNTIME_ATTRIBUTES_H
