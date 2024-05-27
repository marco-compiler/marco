#ifndef MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICABASE_TD
#define MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICABASE_TD

include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Base Modelica dialect
//===---------------------------------------------------------------------===//

def BaseModelica_Dialect : Dialect {
    let name = "bmodelica";
    let summary = "MLIR Base Modelica dialect.";

    let description = [{
        MLIR Base Modelica dialect.
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
        void registerOperations();
    }];
}

#endif // MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICABASE_TD
