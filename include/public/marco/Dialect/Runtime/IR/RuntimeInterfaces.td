#ifndef MARCO_DIALECTS_RUNTIME_IR_RUNTIMEINTERFACES_TD
#define MARCO_DIALECTS_RUNTIME_IR_RUNTIMEINTERFACES_TD

include "marco/Dialect/Runtime/IR/RuntimeDialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Runtime interface definitions
//===----------------------------------------------------------------------===//

class Runtime_Interface<string name> : OpInterface<name>
{
    let cppNamespace = "::mlir::runtime";
}

#endif // MARCO_DIALECTS_RUNTIME_IR_RUNTIMEINTERFACES_TD
