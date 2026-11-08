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
        The dialect represents Modelica models as described by the Modelica
        Language Specification, before instantiation and flattening.
    }];

    let cppNamespace = "::mlir::modelica";
    let hasConstantMaterializer = 1;
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
        void registerOperations();
    }];
}

#endif // MARCO_DIALECT_MODELICA_IR_DIALECT_TD
