#ifndef MARCO_DIALECT_KINSOL_IR_OPS_H
#define MARCO_DIALECT_KINSOL_IR_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "marco/Dialect/KINSOL/IR/Attributes.h"
#include "marco/Dialect/KINSOL/IR/Types.h"

#define GET_OP_CLASSES
#include "marco/Dialect/KINSOL/IR/KINSOLOps.h.inc"

#endif // MARCO_DIALECT_KINSOL_IR_OPS_H
