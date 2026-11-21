#ifndef MARCO_DIALECT_MODELICA_IR_OPINTERFACES_TD
#define MARCO_DIALECT_MODELICA_IR_OPINTERFACES_TD

include "marco/Dialect/Modelica/IR/Dialect.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Modelica op interface definitions
//===---------------------------------------------------------------------===//

class Modelica_OpInterface<string name>
    : OpInterface<name>
{
    let cppNamespace = "::mlir::modelica";
}

def Modelica_ClassInterface : Modelica_OpInterface<"ClassInterface">
{
    let description = [{
        Interface of the operations representing the definition of a class,
        one for each class restriction. It gives access to the properties that
        don't depend on the restriction.
    }];

    let methods = [
        InterfaceMethod<
            "Get the name of the class.",
            "::llvm::StringRef", "getSymName", (ins),
            [{ return $_op.getSymName(); }]>,
        InterfaceMethod<
            "Get the body of the class.",
            "::mlir::Block *", "getBody", (ins),
            [{ return $_op.getBody(); }]>,
        InterfaceMethod<
            "Get whether the class is declared as replaceable.",
            "bool", "getIsReplaceable", (ins),
            [{ return $_op.getIsReplaceable(); }]>,
        InterfaceMethod<
            "Get whether the class is encapsulated.",
            "bool", "getIsEncapsulated", (ins),
            [{ return $_op.getIsEncapsulated(); }]>,
        InterfaceMethod<
            "Get whether the class is a short class definition.",
            "bool", "getIsShort", (ins),
            [{ return $_op.getIsShort(); }]>,
        InterfaceMethod<
            "Set whether the class is a short class definition.",
            "void", "setIsShort", (ins "bool":$value),
            [{ $_op.setIsShort(value); }]>,
        InterfaceMethod<
            "Get whether the class is a class-extends redeclaration.",
            "bool", "getIsClassExtends", (ins),
            [{ return $_op.getIsClassExtends(); }]>,
        InterfaceMethod<
            "Set whether the class is a class-extends redeclaration.",
            "void", "setIsClassExtends", (ins "bool":$value),
            [{ $_op.setIsClassExtends(value); }]>,
        InterfaceMethod<
            "Get the description string of the class.",
            "::mlir::StringAttr", "getDescriptionAttr", (ins),
            [{ return $_op.getDescriptionAttr(); }]>,
        InterfaceMethod<
            "Set the description string of the class.",
            "void", "setDescriptionAttr", (ins "::mlir::StringAttr":$value),
            [{ $_op.setDescriptionAttr(value); }]>,
        InterfaceMethod<
            "Get the annotation of the class.",
            "::mlir::DictionaryAttr", "getAnnotationAttr", (ins),
            [{ return $_op.getAnnotationAttr(); }]>,
        InterfaceMethod<
            "Set the annotation of the class.",
            "void", "setAnnotationAttr", (ins "::mlir::DictionaryAttr":$value),
            [{ $_op.setAnnotationAttr(value); }]>
    ];
}

#endif // MARCO_DIALECT_MODELICA_IR_OPINTERFACES_TD
