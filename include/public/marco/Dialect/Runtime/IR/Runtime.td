#ifndef MARCO_DIALECT_RUNTIME_IR_RUNTIMEDIALECT_TD
#define MARCO_DIALECT_RUNTIME_IR_RUNTIMEDIALECT_TD

include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Runtime dialect
//===---------------------------------------------------------------------===//

def Runtime_Dialect : Dialect {
    let name = "runtime";
    let summary = "MLIR Runtime dialect.";

    let description = [{
        MLIR Runtime dialect.
    }];

    let cppNamespace = "::mlir::runtime";
    let useDefaultAttributePrinterParser = 1;
    
    let dependentDialects = [
        "modeling::ModelingDialect"
    ];

    let extraClassDeclaration = [{
        void registerAttributes();
    }];
}

#endif // MARCO_DIALECT_RUNTIME_IR_RUNTIMEDIALECT_TD
