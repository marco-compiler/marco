#ifndef MARCO_DIALECT_MODELICA_IR_MODELICATYPES_TD
#define MARCO_DIALECT_MODELICA_IR_MODELICATYPES_TD

include "marco/Dialect/Modelica/IR/Modelica.td"
include "marco/Dialect/Modelica/IR/ModelicaTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===---------------------------------------------------------------------===//
// Modelica type definitions
//===---------------------------------------------------------------------===//

// Base class for Modelica dialect types.
class Modelica_Type<
    string typeName,
    list<Trait> traits = [],
    string baseCppClass = "::mlir::Type">
    : TypeDef<Modelica_Dialect, typeName, traits, baseCppClass>
{
    let mnemonic = ?;
}

#endif // MARCO_DIALECT_MODELICA_IR_MODELICATYPES_TD
