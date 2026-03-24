#ifndef MARCO_DIALECT_BASEMODELICA_IR_ENUMS_TD
#define MARCO_DIALECT_BASEMODELICA_IR_ENUMS_TD

include "mlir/IR/EnumAttr.td"

def AssertionLevel : I32EnumAttr<
    "AssertionLevel", "Assertion level",
    [
        I32EnumAttrCase<"Warning", 0, "Warning">,
        I32EnumAttrCase<"Error", 1, "Error">
    ]> {
    let cppNamespace = "::mlir::bmodelica";
}

#endif // MARCO_DIALECT_BASEMODELICA_IR_ENUMS_TD
