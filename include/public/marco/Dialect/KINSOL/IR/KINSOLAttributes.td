#ifndef MARCO_DIALECT_KINSOL_IR_KINSOLATTRIBUTES_TD
#define MARCO_DIALECT_KINSOL_IR_KINSOLATTRIBUTES_TD

include "marco/Dialect/KINSOL/IR/KINSOL.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===---------------------------------------------------------------------===//
// KINSOL attribute definitions
//===---------------------------------------------------------------------===//

class KINSOL_Attr<string name, string attrMnemonic>
    : AttrDef<KINSOL_Dialect, name>
{
    let mnemonic = attrMnemonic;
}

#endif // MARCO_DIALECT_KINSOL_IR_KINSOLATTRIBUTES_TD
