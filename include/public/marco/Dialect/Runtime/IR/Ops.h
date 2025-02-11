#ifndef MARCO_DIALECT_RUNTIME_OPS_H
#define MARCO_DIALECT_RUNTIME_OPS_H

#include "marco/Dialect/Modeling/IR/Enums.h"
#include "marco/Dialect/Runtime/IR/Attributes.h"
#include "marco/Dialect/Runtime/IR/Properties.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define GET_OP_CLASSES
#include "marco/Dialect/Runtime/IR/RuntimeOps.h.inc"

#endif // MARCO_DIALECT_RUNTIME_OPS_H
