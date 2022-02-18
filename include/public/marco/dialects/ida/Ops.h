#ifndef MARCO_DIALECTS_IDA_OPS_H
#define MARCO_DIALECTS_IDA_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "marco/dialects/ida/Attributes.h"
#include "marco/dialects/ida/Types.h"

#define GET_OP_CLASSES
#include "marco/dialects/ida/IDA.h.inc"

#endif // MARCO_DIALECTS_IDA_OPS_H
