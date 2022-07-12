#ifndef MARCO_DIALECTS_MODELICA_MODELICAINTERFACES_TD
#define MARCO_DIALECTS_MODELICA_MODELICAINTERFACES_TD

include "marco/Dialect/KINSOL/KINSOLDialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// KINSOL interface definitions
//===----------------------------------------------------------------------===//

class KINSOL_TypeInterface<string name> : TypeInterface<name> {
    let cppNamespace = "::mlir::kinsol";
}

#endif // MARCO_DIALECTS_MODELICA_MODELICAINTERFACES_TD
