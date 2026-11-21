#ifndef MARCO_DIALECT_MODELICA_IR_OPS_H
#define MARCO_DIALECT_MODELICA_IR_OPS_H

#include "marco/Dialect/Modelica/IR/Attributes.h"
#include "marco/Dialect/Modelica/IR/Enums.h"
#include "marco/Dialect/Modelica/IR/OpInterfaces.h"
#include "marco/Dialect/Modelica/IR/Properties.h"
#include "marco/Dialect/Modelica/IR/TypeInference.h"
#include "marco/Dialect/Modelica/IR/Types.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_FWD_DEFINES
#include "marco/Dialect/Modelica/IR/Ops.h.inc"

#define GET_OP_CLASSES
#include "marco/Dialect/Modelica/IR/Ops.h.inc"

namespace mlir::modelica {
/// Get whether a value is a component reference, that is the result of a
/// chain of 'modelica.member' and 'modelica.subscript' operations starting
/// from a 'modelica.ref'. If 'allowDerivative' is true, the chain may also be
/// the operand of a 'modelica.der' operation.
bool isComponentReference(mlir::Value value, bool allowDerivative = false);
} // namespace mlir::modelica

#endif // MARCO_DIALECT_MODELICA_IR_OPS_H
