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

def ConstantMaterializableType
    : Modelica_TypeInterface<"ConstantMaterializableType">
{
    let description = [{
        Interface for a type for which a constant can be materialized.
    }];

    let methods = [
        InterfaceMethod<
            "Materialize a constant of this type.",
            "mlir::Value", "materializeBoolConstant",
            (ins "mlir::OpBuilder&":$builder,
                 "mlir::Location":$loc,
                 "bool":$value)>,
        InterfaceMethod<
            "Materialize a constant of this type.",
            "mlir::Value", "materializeIntConstant",
            (ins "mlir::OpBuilder&":$builder,
                 "mlir::Location":$loc,
                 "int64_t":$value)>,
        InterfaceMethod<
            "Materialize a constant of this type.",
            "mlir::Value", "materializeFloatConstant",
            (ins "mlir::OpBuilder&":$builder,
                 "mlir::Location":$loc,
                 "double":$value)>
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
