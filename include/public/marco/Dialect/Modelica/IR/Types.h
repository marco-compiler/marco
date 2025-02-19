#ifndef MARCO_DIALECT_MODELICA_IR_TYPES_H
#define MARCO_DIALECT_MODELICA_IR_TYPES_H

#include "marco/Dialect/Modelica/IR/TypeInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/Modelica/IR/ModelicaTypes.h.inc"

namespace mlir::modelica {} // namespace mlir::modelica

#endif // MARCO_DIALECT_MODELICA_IR_TYPES_H
