#ifndef MARCO_DIALECT_MODELING_IR_MODELING_TD
#define MARCO_DIALECT_MODELING_IR_MODELING_TD

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

#endif // MARCO_DIALECT_MODELING_IR_MODELING_TD
