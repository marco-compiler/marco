#ifndef MARCO_DIALECT_RUNTIME_IR_RUNTIMETYPES_TD
#define MARCO_DIALECT_RUNTIME_IR_RUNTIMETYPES_TD

include "marco/Dialect/Runtime/IR/Runtime.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===---------------------------------------------------------------------===//
// Runtime type definitions
//===---------------------------------------------------------------------===//

// Base class for Base Modelica dialect types.
class Runtime_Type<
    string typeName,
    list<Trait> traits = [],
    string baseCppClass = "::mlir::Type">
    : TypeDef<Runtime_Dialect, typeName, traits, baseCppClass>
{
    let mnemonic = ?;
}

def StringType : Runtime_Type<"String">
{
    let summary = "String.";
    let mnemonic = "string";
    let description = "Type associated to a string.";
}

#endif // MARCO_DIALECT_RUNTIME_IR_RUNTIMETYPES_TD
