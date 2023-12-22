#ifndef MARCO_DIALECTS_SUNDIALS_SUNDIALSDIALECT_TD
#define MARCO_DIALECTS_SUNDIALS_SUNDIALSDIALECT_TD

include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// SUNDIALS dialect
//===---------------------------------------------------------------------===//

def SUNDIALS_Dialect : Dialect
{
    let name = "sundials";
    let summary = "MLIR SUNDIALS dialect.";

    let description = [{
        MLIR SUNDIALS dialect.
    }];

    let cppNamespace = "::mlir::sundials";
    let useDefaultTypePrinterParser = 1;
}

#endif // MARCO_DIALECTS_SUNDIALS_SUNDIALSDIALECT_TD
