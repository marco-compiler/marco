#ifndef MARCO_DIALECT_IDA_IR_TYPES_TD
#define MARCO_DIALECT_IDA_IR_TYPES_TD

include "marco/Dialect/IDA/IR/Dialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

//===---------------------------------------------------------------------===//
// IDA type definitions
//===---------------------------------------------------------------------===//

class IDA_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<IDA_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def InstanceType : IDA_Type<"Instance", "instance">
{
    let summary = "IDA instance.";

    let description = [{
        The type represents an instance of IDA.
    }];
}

def EquationType : IDA_Type<"Equation", "equation">
{
    let summary = "IDA equation.";

    let description = [{
        The type represents an equation that is managed by IDA.
    }];
}

def VariableType : IDA_Type<"Variable", "variable",
    [MemRefElementTypeInterface,
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface, [
        "getTypeSizeInBits", "getABIAlignment"]>]>
{
    let summary = "IDA variable.";

    let description = [{
        The type represents a variable that is managed by IDA.

        IDA variables can be stored into memrefs, so that the handles obtained
        when registering the variables can be shared among different functions.
    }];
}

#endif // MARCO_DIALECT_IDA_IR_TYPES_TD
