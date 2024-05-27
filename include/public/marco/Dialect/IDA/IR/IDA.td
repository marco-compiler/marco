#ifndef MARCO_DIALECT_IDA_IR_IDA_TD
#define MARCO_DIALECT_IDA_IR_IDA_TD

include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// IDA dialect
//===---------------------------------------------------------------------===//

def IDA_Dialect : Dialect {
    let name = "ida";
    let summary = "MLIR IDA dialect.";

    let description = [{
        MLIR IDA dialect.
    }];

    let cppNamespace = "::mlir::ida";
    let useDefaultTypePrinterParser = 1;

    let dependentDialects = [
        "sundials::SUNDIALSDialect"
    ];
}

#endif // MARCO_DIALECT_IDA_IR_IDA_TD
