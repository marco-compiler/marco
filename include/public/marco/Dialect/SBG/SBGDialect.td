#ifndef MARCO_DIALECTS_SBG_SBGDIALECT_TD
#define MARCO_DIALECTS_SBG_SBGDIALECT_TD

include "mlir/IR/OpBase.td"

def SBG_Dialect : Dialect {
    let name = "sbg";
    let summary = "MLIR SBG dialect.";

    let description = [{
        MLIR SBG dialect.
    }];

    let cppNamespace = "::mlir::sbg";
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];
}

#endif // MARCO_DIALECTS_SBG_SBGDIALECT_TD
