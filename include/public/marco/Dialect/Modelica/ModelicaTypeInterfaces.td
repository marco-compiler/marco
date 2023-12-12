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

def ZeroMaterializableType : Modelica_TypeInterface<"ZeroMaterializableType">
{
    let description = [{
        Interface for a type for which a zero-valued constant can be
        materialized.
    }];

    let methods = [
        InterfaceMethod<
            "Materialize a zero-valued constant of this type.",
            "mlir::Value", "materializeZeroValuedConstant",
            (ins "mlir::OpBuilder&":$builder, "mlir::Location":$loc)>
    ];
}

def OneMaterializableType : Modelica_TypeInterface<"OneMaterializableType">
{
    let description = [{
        Interface for a type for which a one-valued constant can be
        materialized.
    }];

    let methods = [
        InterfaceMethod<
            "Materialize a one-valued constant of this type.",
            "mlir::Value", "materializeOneValuedConstant",
            (ins "mlir::OpBuilder&":$builder, "mlir::Location":$loc)>
    ];
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
