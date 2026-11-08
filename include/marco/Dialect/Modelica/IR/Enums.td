#ifndef MARCO_DIALECT_MODELICA_IR_ENUMS_TD
#define MARCO_DIALECT_MODELICA_IR_ENUMS_TD

include "marco/Dialect/Modelica/IR/Dialect.td"
include "mlir/IR/EnumAttr.td"

//===---------------------------------------------------------------------===//
// Modelica enum definitions
//===---------------------------------------------------------------------===//

def Modelica_ReductionKind : I32EnumAttr<
    "ReductionKind",
    "Kind of reduction expression",
    [
        I32EnumAttrCase<"sum", 0>,
        I32EnumAttrCase<"product", 1>,
        I32EnumAttrCase<"min", 2>,
        I32EnumAttrCase<"max", 3>
    ]>
{
    let cppNamespace = "::mlir::modelica";
}

def Modelica_ClassKind : I32EnumAttr<
    "ClassKind",
    "Kind of class",
    [
        I32EnumAttrCase<"class_", 0, "class">,
        I32EnumAttrCase<"model", 1>,
        I32EnumAttrCase<"record", 2>,
        I32EnumAttrCase<"operator_record", 3>,
        I32EnumAttrCase<"block", 4>,
        I32EnumAttrCase<"connector", 5>,
        I32EnumAttrCase<"expandable_connector", 6>,
        I32EnumAttrCase<"type", 7>,
        I32EnumAttrCase<"package", 8>,
        I32EnumAttrCase<"function", 9>,
        I32EnumAttrCase<"operator_", 10, "operator">,
        I32EnumAttrCase<"operator_function", 11>
    ]>
{
    let cppNamespace = "::mlir::modelica";
}

def Modelica_Variability : I32EnumAttr<
    "Variability",
    "Declared variability of a component",
    [
        I32EnumAttrCase<"discrete", 0>,
        I32EnumAttrCase<"parameter", 1>,
        I32EnumAttrCase<"constant", 2>
    ]>
{
    let cppNamespace = "::mlir::modelica";
}

def Modelica_Causality : I32EnumAttr<
    "Causality",
    "Causality of a component",
    [
        I32EnumAttrCase<"input", 0>,
        I32EnumAttrCase<"output", 1>
    ]>
{
    let cppNamespace = "::mlir::modelica";
}

def Modelica_ConnectorKind : I32EnumAttr<
    "ConnectorKind",
    "Kind of connector variable",
    [
        I32EnumAttrCase<"flow", 0>,
        I32EnumAttrCase<"stream", 1>
    ]>
{
    let cppNamespace = "::mlir::modelica";
}

def Modelica_DimensionKind : I32EnumAttr<
    "DimensionKind",
    "Kind of array dimension in a declaration",
    [
        I32EnumAttrCase<"colon", 0>,
        I32EnumAttrCase<"fixed", 1>,
        I32EnumAttrCase<"expression", 2>,
        I32EnumAttrCase<"boolean", 3>,
        I32EnumAttrCase<"enumeration", 4>
    ]>
{
    let cppNamespace = "::mlir::modelica";
}

#endif // MARCO_DIALECT_MODELICA_IR_ENUMS_TD
