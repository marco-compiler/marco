#ifndef MARCO_DIALECTS_MODELICA_MODELICAINTERFACES_TD
#define MARCO_DIALECTS_MODELICA_MODELICAINTERFACES_TD

include "marco/dialects/ida/IDADialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IDA interface definitions
//===----------------------------------------------------------------------===//

class IDA_TypeInterface<string name> : TypeInterface<name> {
    let cppNamespace = "::mlir::ida";
}

#endif // MARCO_DIALECTS_MODELICA_MODELICAINTERFACES_TD
