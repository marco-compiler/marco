#ifndef MARCO_DIALECTS_SUNDIALS_SUNDIALSTYPES_TD
#define MARCO_DIALECTS_SUNDIALS_SUNDIALSTYPES_TD

include "marco/Dialect/SUNDIALS/SUNDIALSDialect.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// SUNDIALS type definitions
//===---------------------------------------------------------------------===//

class SUNDIALS_Type<string name, string typeMnemonic>
    : TypeDef<SUNDIALS_Dialect, name>
{
    let mnemonic = typeMnemonic;
}

def EquationType : SUNDIALS_Type<"Equation", "equation">
{
    let summary = "SUNDIALS equation.";

    let description = [{
        The type represents an equation that is managed by SUNDIALS.
    }];
}

def VariableType : SUNDIALS_Type<"Variable", "variable">
{
    let summary = "SUNDIALS variable.";

    let description = [{
        The type represents a variable that is managed by SUNDIALS.
    }];
}

#endif // MARCO_DIALECTS_SUNDIALS_SUNDIALSTYPES_TD
