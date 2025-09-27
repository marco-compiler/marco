#ifndef MARCO_DIALECT_MODELICA_IR_MODELICAATTRIBUTES_TD
#define MARCO_DIALECT_MODELICA_IR_MODELICAATTRIBUTES_TD

include "marco/Dialect/Modelica/IR/Modelica.td"
include "marco/Dialect/Modelica/IR/ModelicaAttrInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===---------------------------------------------------------------------===//
// Modelica attribute definitions
//===---------------------------------------------------------------------===//

class Modelica_Attr<string name,
                    list<Trait> traits = [],
                    string baseCppClass = "::mlir::Attribute">
    : AttrDef<Modelica_Dialect, name, traits, baseCppClass>
{
    let mnemonic = ?;
}

#endif // MARCO_DIALECT_MODELICA_IR_MODELICAATTRIBUTES_TD
