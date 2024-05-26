#ifndef MARCO_DIALECTS_KINSOL_IR_KINSOLINTERFACES_TD
#define MARCO_DIALECTS_KINSOL_IR_KINSOLINTERFACES_TD

include "marco/Dialect/KINSOL/IR/KINSOLDialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// KINSOL interface definitions
//===----------------------------------------------------------------------===//

class KINSOL_TypeInterface<string name> : TypeInterface<name>
{
    let cppNamespace = "::mlir::kinsol";
}

#endif // MARCO_DIALECTS_KINSOL_IR_KINSOLINTERFACES_TD
