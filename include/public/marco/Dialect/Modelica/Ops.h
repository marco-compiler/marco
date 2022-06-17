#ifndef MARCO_DIALECTS_MODELICA_MODELICAOPS_H
#define MARCO_DIALECTS_MODELICA_MODELICAOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "marco/Dialect/Modelica/Attributes.h"
#include "marco/Dialect/Modelica/Types.h"

#define GET_OP_CLASSES
#include "marco/Dialect/Modelica/Modelica.h.inc"

#endif // MARCO_DIALECTS_MODELICA_MODELICAOPS_H
