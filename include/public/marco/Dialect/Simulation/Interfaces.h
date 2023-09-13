#ifndef MARCO_DIALECTS_SIMULATION_INTERFACES_H
#define MARCO_DIALECTS_SIMULATION_INTERFACES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::simulation::function_interface_impl
{
  // TODO remove
  /// Return the name of the attribute used for function types.
  inline llvm::StringRef getTypeAttrName()
  {
    return "function_type";
  }
}

#include "marco/Dialect/Simulation/SimulationInterfaces.h.inc"

#endif // MARCO_DIALECTS_SIMULATION_INTERFACES_H
