#ifndef MARCO_DIALECTS_SIMULATION_OPS_H
#define MARCO_DIALECTS_SIMULATION_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "marco/Dialect/Simulation/Attributes.h"
#include "marco/Dialect/Simulation/Types.h"

#define GET_OP_CLASSES
#include "marco/Dialect/Simulation/Simulation.h.inc"

#endif // MARCO_DIALECTS_SIMULATION_OPS_H
