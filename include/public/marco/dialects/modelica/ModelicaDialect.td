#ifndef MARCO_DIALECTS_MODELICA_MODELICA_DIALECT_TD
#define MARCO_DIALECTS_MODELICA_MODELICA_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Modelica_Dialect : Dialect {
    let name = "modelica";
    let summary = "MLIR Modelica dialect.";

    let description = [{
        MLIR Modelica dialect.
    }];

    let cppNamespace = "::mlir::modelica";
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

class Modelica_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Modelica_Dialect, mnemonic, traits>;

#endif // MARCO_DIALECTS_MODELICA_MODELICA_DIALECT_TD
