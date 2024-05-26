#ifndef MARCO_DIALECTS_IDA_IR_IDATYPES_TD
#define MARCO_DIALECTS_IDA_IR_IDATYPES_TD

include "marco/Dialect/IDA/IR/IDADialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IDA type definitions
//===----------------------------------------------------------------------===//

class IDA_Type<string name, string typeMnemonic>
    : TypeDef<IDA_Dialect, name> {
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

def VariableType : IDA_Type<"Variable", "variable">
{
    let summary = "IDA variable.";

    let description = [{
        The type represents a variable that is managed by IDA.
    }];
}

#endif // MARCO_DIALECTS_IDA_IR_IDATYPES_TD
