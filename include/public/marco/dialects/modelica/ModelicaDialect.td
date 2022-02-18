#ifndef MARCO_DIALECTS_MODELICA_MODELICADIALECT_TD
#define MARCO_DIALECTS_MODELICA_MODELICADIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Modelica dialect
//===----------------------------------------------------------------------===//

def Modelica_Dialect : Dialect {
    let name = "modelica";
    let summary = "MLIR Modelica dialect.";

    let description = [{
        MLIR Modelica dialect.
    }];

    let cppNamespace = "::mlir::modelica";
}

#endif // MARCO_DIALECTS_MODELICA_MODELICADIALECT_TD
