#ifndef MARCO_DIALECTS_SUNDIALS_SUNDIALSINTERFACES_TD
#define MARCO_DIALECTS_SUNDIALS_SUNDIALSINTERFACES_TD

include "marco/Dialect/SUNDIALS/SUNDIALSDialect.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// SUNDIALS interface definitions
//===---------------------------------------------------------------------===//

class SUNDIALS_TypeInterface<string name> : TypeInterface<name>
{
    let cppNamespace = "::mlir::sundials";
}

#endif // MARCO_DIALECTS_SUNDIALS_SUNDIALSINTERFACES_TD
