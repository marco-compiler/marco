#ifndef MARCO_DIALECTS_IDA_IDAATTRIBUTES_TD
#define MARCO_DIALECTS_IDA_IDAATTRIBUTES_TD

include "marco/Dialect/IDA/IDADialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IDA attribute definitions
//===----------------------------------------------------------------------===//

class IDA_Attr<string name, string attrMnemonic>
    : AttrDef<IDA_Dialect, name> {
    let mnemonic = attrMnemonic;
}

#endif // MARCO_DIALECTS_IDA_IDAATTRIBUTES_TD
