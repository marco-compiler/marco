include "mlir/IR/EnumAttr.td"

def EquationDependencyKind : I32EnumAttr<
    "EquationDependencyKind", "Kind of dependency between points of the iteration space of the equation",
    [
        I32EnumAttrCase<"Independent", 0, "Completely independent indices">,
        I32EnumAttrCase<"BackwardNeighbour", 1, "Backward dependency on neighbour indices">,
        I32EnumAttrCase<"Other", 2, "Other kind of dependencies">
    ]> {
    let cppNamespace = "::mlir::bmodelica";
}
