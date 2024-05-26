#ifndef MARCO_DIALECTS_KINSOL_IR_KINSOLTYPES_TD
#define MARCO_DIALECTS_KINSOL_IR_KINSOLTYPES_TD

include "marco/Dialect/KINSOL/IR/KINSOLDialect.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// KINSOL type definitions
//===---------------------------------------------------------------------===//

class KINSOL_Type<string name, string typeMnemonic>
    : TypeDef<KINSOL_Dialect, name>
{
    let mnemonic = typeMnemonic;
}

def InstanceType : KINSOL_Type<"Instance", "instance">
{
    let summary = "KINSOL instance.";

    let description = [{
        The type represents an instance of KINSOL.
    }];
}

def EquationType : KINSOL_Type<"Equation", "equation">
{
    let summary = "KINSOL equation.";

    let description = [{
        The type represents an equation that is managed by KINSOL.
    }];
}

def VariableType : KINSOL_Type<"Variable", "variable">
{
    let summary = "KINSOL variable.";

    let description = [{
        The type represents a variable that is managed by KINSOL.
    }];
}

#endif // MARCO_DIALECTS_KINSOL_IR_KINSOLTYPES_TD
