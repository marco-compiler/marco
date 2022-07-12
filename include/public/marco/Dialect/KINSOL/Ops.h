#ifndef MARCO_DIALECTS_KINSOL_OPS_H
#define MARCO_DIALECTS_KINSOL_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "marco/Dialect/KINSOL/Attributes.h"
#include "marco/Dialect/KINSOL/Types.h"

#define GET_OP_CLASSES
#include "marco/Dialect/KINSOL/KINSOL.h.inc"

#endif // MARCO_DIALECTS_KINSOL_OPS_H
