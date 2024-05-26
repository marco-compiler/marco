#ifndef MARCO_DIALECTS_MODELING_IR_MODELINGTYPES_TD
#define MARCO_DIALECTS_MODELING_IR_MODELINGTYPES_TD

include "marco/Dialect/Modeling/IR/ModelingDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===---------------------------------------------------------------------===//
// Modeling type definitions
//===---------------------------------------------------------------------===//

// Base class for Modelica dialect types.
class Modeling_Type<
    string name,
    list<Trait> traits = [],
    string baseCppClass = "::mlir::Type">
    : TypeDef<Modeling_Dialect, name, traits, baseCppClass>
{
    let mnemonic = ?;
}

#endif // MARCO_DIALECTS_MODELING_IR_MODELINGTYPES_TD
