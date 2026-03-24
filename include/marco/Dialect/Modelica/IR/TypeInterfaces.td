#ifndef MARCO_DIALECT_MODELICA_IR_TYPEINTERFACES_TD
#define MARCO_DIALECT_MODELICA_IR_TYPEINTERFACES_TD

include "marco/Dialect/Modelica/IR/Dialect.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Modelica type interface definitions
//===---------------------------------------------------------------------===//

class Modelica_TypeInterface<string name>
    : TypeInterface<name>
{
    let cppNamespace = "::mlir::modelica";
}

#endif // MARCO_DIALECT_MODELICA_IR_TYPEINTERFACES_TD
