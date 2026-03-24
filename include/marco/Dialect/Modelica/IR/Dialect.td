#ifndef MARCO_DIALECT_MODELICA_IR_DIALECT_TD
#define MARCO_DIALECT_MODELICA_IR_DIALECT_TD

include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Modelica dialect
//===---------------------------------------------------------------------===//

def Modelica_Dialect : Dialect {
    let name = "modelica";
    let summary = "MLIR Modelica dialect.";

    let description = [{
        MLIR Modelica dialect.
    }];

    let cppNamespace = "::mlir::modelica";
    let useDefaultAttributePrinterParser = 0;
    let useDefaultTypePrinterParser = 0;

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
        void registerOperations();
    }];
}

#endif // MARCO_DIALECT_MODELICA_IR_DIALECT_TD
