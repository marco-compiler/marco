#ifndef MARCO_DIALECTS_BASEMODELICA_IR_BASEMODELICATYPEINTERFACES_TD
#define MARCO_DIALECTS_BASEMODELICA_IR_BASEMODELICATYPEINTERFACES_TD

include "marco/Dialect/BaseModelica/IR/BaseModelicaDialect.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Base Modelica type interface definitions
//===---------------------------------------------------------------------===//

class BaseModelica_TypeInterface<string name>
    : TypeInterface<name>
{
    let cppNamespace = "::mlir::bmodelica";
}

def ElementaryType : BaseModelica_TypeInterface<"ElementaryType">
{
    let description = [{
        Interface for an elementary type.
    }];
}

def ConstantMaterializableTypeInterface
    : BaseModelica_TypeInterface<"ConstantMaterializableTypeInterface">
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

def DerivableTypeInterface : BaseModelica_TypeInterface<"DerivableTypeInterface">
{
    let description = [{
        Interface for a type for which can undergo differentiation.
    }];

    let methods = [
        InterfaceMethod<
            "Get the derived type.",
            "mlir::FailureOr<mlir::Type>", "derive",
            (ins)>
    ];
}

def IterableTypeInterface : BaseModelica_TypeInterface<"IterableTypeInterface">
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

#endif // MARCO_DIALECTS_BASEMODELICA_IR_BASEMODELICATYPEINTERFACES_TD
