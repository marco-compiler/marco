#ifndef MARCO_DIALECT_MODELING_IR_MODELINGATTRIBUTES_TD
#define MARCO_DIALECT_MODELING_IR_MODELINGATTRIBUTES_TD

include "marco/Dialect/Modeling/IR/Modeling.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===---------------------------------------------------------------------===//
// Modelica attribute definitions
//===---------------------------------------------------------------------===//

class Modeling_Attr<string name, list<Trait> traits = []>
    : AttrDef<Modeling_Dialect, name, traits>
{
    let mnemonic = ?;
}

class RangeParameter<string desc>
    : AttrOrTypeParameter<"::mlir::modeling::Range", desc,
                          "const ::mlir::modeling::Range&">;

def Modeling_RangeAttr : Modeling_Attr<"Range">
{
    let summary = "Range";

    let description = [{
        Attribute representing a range of indices.
    }];

    let mnemonic = "range";
    let parameters = (ins RangeParameter<"Range">:$value);
    let assemblyFormat = "$value";
}

class MultidimensionalRangeParameter<string desc>
    : AttrOrTypeParameter<"::mlir::modeling::MultidimensionalRange", desc,
                          "const ::mlir::modeling::MultidimensionalRange&">;

def Modeling_MultidimensionalRangeAttr : Modeling_Attr<"MultidimensionalRange">
{
    let summary = "Multidimensional range";

    let description = [{
        Attribute representing a multidimensional range of indices.
    }];

    let mnemonic = "multidim_range";

    let parameters = (ins
        MultidimensionalRangeParameter<"Multidimensional range">:$value);

    let assemblyFormat = "$value";
}

#endif // MARCO_DIALECT_MODELING_IR_MODELINGATTRIBUTES_TD
