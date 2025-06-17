include "mlir/IR/EnumAttr.td"

def AssertionLevel : I32EnumAttr<
    "AssertionLevel", "Assertion level",
    [
        I32EnumAttrCase<"Warning", 0, "Warning">,
        I32EnumAttrCase<"Error", 1, "Error">
    ]> {
    let cppNamespace = "::mlir::bmodelica";
}
