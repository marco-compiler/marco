#ifndef MARCO_DIALECTS_RUNTIME_TYPES_H
#define MARCO_DIALECTS_RUNTIME_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/Runtime/RuntimeTypes.h.inc"

#endif // MARCO_DIALECTS_RUNTIME_TYPES_H
