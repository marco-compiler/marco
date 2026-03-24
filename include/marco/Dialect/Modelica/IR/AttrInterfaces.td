#ifndef MARCO_DIALECT_MODELICA_IR_ATTRINTERFACES_TD
#define MARCO_DIALECT_MODELICA_IR_ATTRINTERFACES_TD

include "marco/Dialect/Modelica/IR/Dialect.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Modelica attribute interface definitions
//===---------------------------------------------------------------------===//

class Modelica_AttrInterface<string name>
    : AttrInterface<name>
{
    let cppNamespace = "::mlir::modelica";
}

#endif // MARCO_DIALECT_MODELICA_IR_ATTRINTERFACES_TD
