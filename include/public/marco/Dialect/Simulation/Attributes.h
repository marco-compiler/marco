#ifndef MARCO_DIALECTS_SIMULATION_ATTRIBUTES_H
#define MARCO_DIALECTS_SIMULATION_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Hashing.h"

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/Simulation/SimulationAttributes.h.inc"

#endif // MARCO_DIALECTS_SIMULATION_ATTRIBUTES_H
