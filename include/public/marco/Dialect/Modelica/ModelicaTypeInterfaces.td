#ifndef MARCO_DIALECTS_MODELICA_MODELICATYPEINTERFACES_TD
#define MARCO_DIALECTS_MODELICA_MODELICATYPEINTERFACES_TD

include "marco/Dialect/Modelica/ModelicaDialect.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Modelica type interface definitions
//===---------------------------------------------------------------------===//

class Modelica_TypeInterface<string name>
    : TypeInterface<name>
{
    let cppNamespace = "::mlir::modelica";
}

def ElementaryType : Modelica_TypeInterface<"ElementaryType">
{
    let description = [{
        Interface for an elementary type.
    }];
}

def IterableTypeInterface : Modelica_TypeInterface<"IterableTypeInterface">
{
    let description = [{
        Interface for an iterable type.
    }];

    let methods = [
        InterfaceMethod<
            "Get the type of the induction.",
            "mlir::Type", "getInductionType"
        >
    ];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICATYPEINTERFACES_TD
