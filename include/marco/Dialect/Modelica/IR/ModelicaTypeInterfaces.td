#ifndef MARCO_DIALECT_MODELICA_IR_MODELICATYPEINTERFACES_TD
#define MARCO_DIALECT_MODELICA_IR_MODELICATYPEINTERFACES_TD

include "marco/Dialect/Modelica/IR/Modelica.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Modelica type interface definitions
//===---------------------------------------------------------------------===//

class Modelica_TypeInterface<string name>
    : TypeInterface<name>
{
    let cppNamespace = "::mlir::modelica";
}

#endif // MARCO_DIALECT_MODELICA_IR_MODELICATYPEINTERFACES_TD
