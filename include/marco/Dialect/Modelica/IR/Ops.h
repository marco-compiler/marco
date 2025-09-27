#ifndef MARCO_DIALECT_MODELICA_IR_OPS_H
#define MARCO_DIALECT_MODELICA_IR_OPS_H

#include "marco/Dialect/Modelica/IR/Attributes.h"
#include "marco/Dialect/Modelica/IR/OpInterfaces.h"
#include "marco/Dialect/Modelica/IR/Properties.h"
#include "marco/Dialect/Modelica/IR/Types.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_FWD_DEFINES
#include "marco/Dialect/Modelica/IR/ModelicaOps.h.inc"

#define GET_OP_CLASSES
#include "marco/Dialect/Modelica/IR/ModelicaOps.h.inc"

#endif // MARCO_DIALECT_MODELICA_IR_OPS_H
