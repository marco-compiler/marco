#ifndef MARCO_DIALECT_MODELICA_IR_ATTRIBUTES_H
#define MARCO_DIALECT_MODELICA_IR_ATTRIBUTES_H

#include "marco/Dialect/Modelica/IR/AttrInterfaces.h"
#include "marco/Dialect/Modelica/IR/Types.h"
#include "mlir/IR/Attributes.h"

namespace mlir::modelica {} // namespace mlir::modelica

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/Modelica/IR/ModelicaAttributes.h.inc"

#endif // MARCO_DIALECT_MODELICA_IR_ATTRIBUTES_H
