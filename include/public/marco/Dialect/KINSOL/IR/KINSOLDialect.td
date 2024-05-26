#ifndef MARCO_DIALECTS_KINSOL_IR_KINSOLDIALECT_TD
#define MARCO_DIALECTS_KINSOL_IR_KINSOLDIALECT_TD

include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// KINSOL dialect
//===---------------------------------------------------------------------===//

def KINSOL_Dialect : Dialect
{
    let name = "kinsol";
    let summary = "MLIR KINSOL dialect.";

    let description = [{
        MLIR KINSOL dialect.
    }];

    let cppNamespace = "::mlir::kinsol";
    let useDefaultTypePrinterParser = 1;

    let dependentDialects = [
        "sundials::SUNDIALSDialect"
    ];
}

#endif // MARCO_DIALECTS_KINSOL_IR_KINSOLDIALECT_TD
