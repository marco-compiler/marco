#ifndef MARCO_DIALECTS_IDA_IDAINTERFACES_TD
#define MARCO_DIALECTS_IDA_IDAINTERFACES_TD

include "marco/Dialect/IDA/IDADialect.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// IDA interface definitions
//===---------------------------------------------------------------------===//

class IDA_TypeInterface<string name> : TypeInterface<name>
{
    let cppNamespace = "::mlir::ida";
}

#endif // MARCO_DIALECTS_BASEMODELICA_IDAINTERFACES_TD
