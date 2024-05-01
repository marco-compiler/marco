#ifndef MARCO_DIALECTS_RUNTIME_RUNTIMETYPES_TD
#define MARCO_DIALECTS_RUNTIME_RUNTIMETYPES_TD

include "marco/Dialect/Runtime/RuntimeDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===---------------------------------------------------------------------===//
// Runtime type definitions
//===---------------------------------------------------------------------===//

// Base class for Runtime dialect types.
class Runtime_Type<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
    : TypeDef<Runtime_Dialect, name, traits, baseCppClass> {
    let mnemonic = ?;
}

#endif // MARCO_DIALECTS_RUNTIME_RUNTIMETYPES_TD
