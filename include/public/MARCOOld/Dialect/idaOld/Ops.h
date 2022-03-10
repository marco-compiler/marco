#ifndef MARCO_DIALECTS_IDA_OPS_H
#define MARCO_DIALECTS_IDA_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "marco/Dialect/IDA/Attributes.h"
#include "marco/Dialect/IDA/Types.h"

#define GET_OP_CLASSES
#include "marco/Dialect/IDA/IDA.h.inc"

#endif // MARCO_DIALECTS_IDA_OPS_H
