#ifndef MARCO_DIALECT_MODELICA_IR_OPINTERFACES_TD
#define MARCO_DIALECT_MODELICA_IR_OPINTERFACES_TD

include "marco/Dialect/Modelica/IR/Dialect.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Modelica op interface definitions
//===---------------------------------------------------------------------===//

class Modelica_OpInterface<string name>
    : OpInterface<name>
{
    let cppNamespace = "::mlir::modelica";
}

#endif // MARCO_DIALECT_MODELICA_IR_OPINTERFACES_TD
