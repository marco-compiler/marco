#ifndef MARCO_DIALECT_MODELICA_IR_ATTRIBUTES_H
#define MARCO_DIALECT_MODELICA_IR_ATTRIBUTES_H

#include "marco/Dialect/Modelica/IR/AttrInterfaces.h"
#include "marco/Dialect/Modelica/IR/Enums.h"
#include "marco/Dialect/Modelica/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/Modelica/IR/Attributes.h.inc"

#endif // MARCO_DIALECT_MODELICA_IR_ATTRIBUTES_H
