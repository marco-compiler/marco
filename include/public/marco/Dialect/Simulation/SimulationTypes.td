#ifndef MARCO_DIALECTS_SIMULATION_SIMULATIONTYPES_TD
#define MARCO_DIALECTS_SIMULATION_SIMULATIONTYPES_TD

include "marco/Dialect/Simulation/SimulationDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===---------------------------------------------------------------------===//
// Simulation type definitions
//===---------------------------------------------------------------------===//

// Base class for Simulation dialect types.
class Simulation_Type<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
    : TypeDef<Simulation_Dialect, name, traits, baseCppClass> {
    let mnemonic = ?;
}

#endif // MARCO_DIALECTS_SIMULATION_SIMULATIONTYPES_TD
