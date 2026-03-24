#ifndef MARCO_DIALECT_IDA_IR_ATTRIBUTES_TD
#define MARCO_DIALECT_IDA_IR_ATTRIBUTES_TD

include "marco/Dialect/IDA/IR/Dialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===---------------------------------------------------------------------===//
// IDA attribute definitions
//===---------------------------------------------------------------------===//

class IDA_Attr<string name, string attrMnemonic>
    : AttrDef<IDA_Dialect, name>
{
    let mnemonic = attrMnemonic;
}

#endif // MARCO_DIALECT_IDA_IR_ATTRIBUTES_TD
