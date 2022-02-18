#ifndef MARCO_DIALECTS_IDA_IDATYPES_TD
#define MARCO_DIALECTS_IDA_IDATYPES_TD

include "marco/dialects/ida/IDADialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IDA type definitions
//===----------------------------------------------------------------------===//

class IDA_Type<string name, string typeMnemonic>
    : TypeDef<IDA_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def InstanceType : IDA_Type<"Instance", "instance"> {
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let printer = [{
        $_printer << "instance";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

#endif // MARCO_DIALECTS_IDA_IDATYPES_TD
