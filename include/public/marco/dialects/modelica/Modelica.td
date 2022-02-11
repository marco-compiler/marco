#ifndef MARCO_DIALECTS_MODELICA_MODELICA_TD
#define MARCO_DIALECTS_MODELICA_MODELICA_TD

include "marco/dialects/modelica/ModelicaAttributes.td"
include "marco/dialects/modelica/ModelicaDialect.td"
include "marco/dialects/modelica/ModelicaInterfaces.td"
include "marco/dialects/modelica/ModelicaTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Modelica_FooOp : Modelica_Op<"foo", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `standalone.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = standalone.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICA_TD
