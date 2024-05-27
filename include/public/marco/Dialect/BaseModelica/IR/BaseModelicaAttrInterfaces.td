#ifndef MARCO_DIALECT_BASEMODELICA_IR_MODELICAATTRINTERFACES_TD
#define MARCO_DIALECT_BASEMODELICA_IR_MODELICAATTRINTERFACES_TD

include "marco/Dialect/BaseModelica/IR/BaseModelica.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Base Modelica attribute interface definitions
//===---------------------------------------------------------------------===//

class BaseModelica_AttrInterface<string name>
    : AttrInterface<name>
{
    let cppNamespace = "::mlir::bmodelica";
}

def RangeAttrInterface : BaseModelica_AttrInterface<"RangeAttrInterface">
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

#endif // MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICAATTRINTERFACES_TD
