#ifndef MARCO_DIALECTS_BASEMODELICA_BASEMODELICADIALECT_TD
#define MARCO_DIALECTS_BASEMODELICA_BASEMODELICADIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Base Modelica dialect
//===----------------------------------------------------------------------===//

def BaseModelica_Dialect : Dialect {
    let name = "bmodelica";
    let summary = "MLIR Base Modelica dialect.";

    let description = [{
        MLIR Modelica dialect.
    }];

    let cppNamespace = "::mlir::bmodelica";
    let hasConstantMaterializer = 1;
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;

    let dependentDialects = [
        "modeling::ModelingDialect"
    ];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];
}

#endif // MARCO_DIALECTS_BASEMODELICA_BASEMODELICADIALECT_TD
