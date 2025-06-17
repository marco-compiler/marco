#ifndef MARCO_DIALECT_SUNDIALS_IR_SUNDIALSATTRIBUTES_TD
#define MARCO_DIALECT_SUNDIALS_IR_SUNDIALSATTRIBUTES_TD

include "marco/Dialect/SUNDIALS/IR/SUNDIALS.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===---------------------------------------------------------------------===//
// SUNDIALS attribute definitions
//===---------------------------------------------------------------------===//

class SUNDIALS_Attr<string name, string attrMnemonic>
    : AttrDef<SUNDIALS_Dialect, name>
{
    let mnemonic = attrMnemonic;
}

#endif // MARCO_DIALECT_SUNDIALS_IR_SUNDIALSATTRIBUTES_TD
