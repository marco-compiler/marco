#ifndef MARCO_DIALECTS_IDA_IDATYPES_TD
#define MARCO_DIALECTS_IDA_IDATYPES_TD

include "marco/Dialect/IDA/IDADialect.td"
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

    let printer = [{
        $_printer << "instance";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def EquationType : IDA_Type<"Equation", "equation">
{
    let summary = "IDA equation.";

    let description = [{
        The type represents an equation that is managed by IDA.
    }];

    let printer = [{
        $_printer << "equation";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def VariableType : IDA_Type<"Variable", "variable">
{
    let summary = "IDA variable.";

    let description = [{
        The type represents a variable that is managed by IDA.
    }];

    let printer = [{
        $_printer << "variable";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

#endif // MARCO_DIALECTS_IDA_IDATYPES_TD
