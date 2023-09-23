#ifndef MARCO_DIALECTS_MODELICA_MODELICAATTRINTERFACES_TD
#define MARCO_DIALECTS_MODELICA_MODELICAATTRINTERFACES_TD

include "marco/Dialect/Modelica/ModelicaDialect.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Modelica attribute interface definitions
//===---------------------------------------------------------------------===//

class Modelica_AttrInterface<string name>
    : AttrInterface<name>
{
    let cppNamespace = "::mlir::modelica";
}

def RangeAttrInterface : Modelica_AttrInterface<"RangeAttrInterface">
{
    let description = [{
        Interface for a range.
    }];

    let methods = [
        InterfaceMethod<
            "Get the number of elements of the range.",
            "int64_t", "getNumOfElements"
        >
    ];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICAATTRINTERFACES_TD
