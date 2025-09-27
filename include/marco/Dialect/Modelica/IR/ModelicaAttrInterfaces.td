#ifndef MARCO_DIALECT_MODELICA_IR_MODELICAATTRINTERFACES_TD
#define MARCO_DIALECT_MODELICA_IR_MODELICAATTRINTERFACES_TD

include "marco/Dialect/Modelica/IR/Modelica.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Modelica attribute interface definitions
//===---------------------------------------------------------------------===//

class Modelica_AttrInterface<string name>
    : AttrInterface<name>
{
    let cppNamespace = "::mlir::modelica";
}

#endif // MARCO_DIALECT_MODELICA_IR_MODELICAATTRINTERFACES_TD
