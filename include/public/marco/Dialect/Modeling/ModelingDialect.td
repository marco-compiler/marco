#ifndef MARCO_DIALECTS_MODELING_MODELINGDIALECT_TD
#define MARCO_DIALECTS_MODELING_MODELINGDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Modelica dialect
//===----------------------------------------------------------------------===//

def Modeling_Dialect : Dialect
{
    let name = "modeling";
    let summary = "MLIR Modeling dialect.";

    let description = [{
        MLIR Modeling dialect.
    }];

    let cppNamespace = "::mlir::modeling";
    let useDefaultAttributePrinterParser = 1;

    let extraClassDeclaration = [{
        void registerAttributes();
    }];
}

#endif // MARCO_DIALECTS_MODELING_MODELINGDIALECT_TD
