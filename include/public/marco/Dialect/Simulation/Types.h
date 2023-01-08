#ifndef MARCO_DIALECTS_SIMULATION_TYPES_H
#define MARCO_DIALECTS_SIMULATION_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/SubElementInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/Simulation/SimulationTypes.h.inc"

#endif // MARCO_DIALECTS_SIMULATION_TYPES_H
