#ifndef MARCO_DIALECT_MODELICA_IR_TYPES_TD
#define MARCO_DIALECT_MODELICA_IR_TYPES_TD

include "marco/Dialect/Modelica/IR/Dialect.td"
include "marco/Dialect/Modelica/IR/TypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===---------------------------------------------------------------------===//
// Modelica type definitions
//===---------------------------------------------------------------------===//

// Base class for Modelica dialect types.
class Modelica_Type<
    string typeName,
    list<Trait> traits = [],
    string baseCppClass = "::mlir::Type">
    : TypeDef<Modelica_Dialect, typeName, traits, baseCppClass>
{
    let mnemonic = ?;
}

def Modelica_BooleanType : Modelica_Type<"Boolean">
{
    let summary = "Boolean type";
    let mnemonic = "bool";

    let description = [{
        Type representing the Modelica Boolean predefined type.
    }];
}

def Modelica_IntegerType : Modelica_Type<"Integer">
{
    let summary = "Integer type";
    let mnemonic = "int";

    let description = [{
        Type representing the Modelica Integer predefined type.

        The bit-width of the type is intentionally left unspecified, and it is
        decided when converting to lower level dialects.
    }];
}

def Modelica_RealType : Modelica_Type<"Real">
{
    let summary = "Real type";
    let mnemonic = "real";

    let description = [{
        Type representing the Modelica Real predefined type.

        The precision of the type is intentionally left unspecified, and it is
        decided when converting to lower level dialects.
    }];
}

def Modelica_StringType : Modelica_Type<"String">
{
    let summary = "String type";
    let mnemonic = "string";

    let description = [{
        Type representing the Modelica String predefined type.
    }];
}

def Modelica_EnumerationType : Modelica_Type<"Enumeration">
{
    let summary = "Enumeration type";
    let mnemonic = "enum";

    let description = [{
        Type representing a value of an enumeration type. The type refers to the
        class declaring the enumeration by means of its symbol path.
    }];

    let parameters = (ins "::mlir::SymbolRefAttr":$name);
    let assemblyFormat = "`<` $name `>`";
}

def Modelica_ObjectType : Modelica_Type<"Object">
{
    let summary = "Object type";
    let mnemonic = "object";

    let description = [{
        Type representing an instance of a non-predefined class, such as a
        record, a connector or a model. The type refers to the class by means of
        its absolute symbol path. As replaceable elements may be redeclared, the
        referred class is not necessarily the class of the instance after
        instantiation, and objects of different classes are considered
        compatible until then.
    }];

    let parameters = (ins "::mlir::SymbolRefAttr":$className);
    let assemblyFormat = "`<` $className `>`";
}

#endif // MARCO_DIALECT_MODELICA_IR_TYPES_TD
