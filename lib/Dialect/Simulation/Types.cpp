#include "marco/Dialect/Simulation/Types.h"
#include "marco/Dialect/Simulation/SimulationDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::simulation;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/Simulation/SimulationTypes.cpp.inc"
