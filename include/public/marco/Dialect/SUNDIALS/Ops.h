#ifndef MARCO_DIALECTS_SUNDIALS_OPS_H
#define MARCO_DIALECTS_SUNDIALS_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "marco/Dialect/SUNDIALS/Attributes.h"
#include "marco/Dialect/SUNDIALS/Types.h"

#define GET_OP_CLASSES
#include "marco/Dialect/SUNDIALS/SUNDIALS.h.inc"

#endif // MARCO_DIALECTS_SUNDIALS_OPS_H
