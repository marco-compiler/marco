#ifndef MARCO_DIALECTS_SBG_OPS_H
#define MARCO_DIALECTS_SBG_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

#include "marco/Dialect/SBG/Attributes.h"
#include "marco/Dialect/SBG/Types.h"

#define GET_OP_CLASSES
#include "marco/Dialect/SBG/SBG.h.inc"

#endif // MARCO_DIALECTS_SBG_OPS_H
