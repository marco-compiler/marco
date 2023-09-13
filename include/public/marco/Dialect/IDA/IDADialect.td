#ifndef MARCO_DIALECTS_IDA_IDADIALECT_TD
#define MARCO_DIALECTS_IDA_IDADIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IDA dialect
//===----------------------------------------------------------------------===//

def IDA_Dialect : Dialect {
    let name = "ida";
    let summary = "MLIR IDA dialect.";

    let description = [{
        MLIR IDA dialect.
    }];

    let cppNamespace = "::mlir::ida";
    let useDefaultTypePrinterParser = 1;
}

#endif // MARCO_DIALECTS_IDA_IDADIALECT_TD
