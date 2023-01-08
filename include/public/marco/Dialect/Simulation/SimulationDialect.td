#ifndef MARCO_DIALECTS_SIMULATION_SIMULATIONDIALECT_TD
#define MARCO_DIALECTS_SIMULATION_SIMULATIONDIALECT_TD

include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Simulation dialect
//===---------------------------------------------------------------------===//

def Simulation_Dialect : Dialect {
    let name = "simulation";
    let summary = "MLIR Simulation dialect.";

    let description = [{
        MLIR Simulation dialect.
    }];

    let cppNamespace = "::mlir::simulation";
    let useDefaultAttributePrinterParser = 1;
    let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;

    let extraClassDeclaration = [{
        void registerAttributes();
    }];
}

#endif // MARCO_DIALECTS_SIMULATION_SIMULATIONDIALECT_TD
