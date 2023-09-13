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
    let hasConstantMaterializer = 1;
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 0;

    let dependentDialects = [
        "modeling::ModelingDialect"
    ];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();

        mlir::Type parseType(mlir::DialectAsmParser& parser) const override;
        void printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const override;
    }];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICADIALECT_TD
