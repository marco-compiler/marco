#ifndef MARCO_DIALECTS_MODELICA_OPINTERFACES_H
#define MARCO_DIALECTS_MODELICA_OPINTERFACES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

#define GET_OP_FWD_DEFINES
#include "marco/Dialect/Modelica/Modelica.h.inc"

#include "marco/Dialect/Modelica/ModelicaOpInterfaces.h.inc"

#endif // MARCO_DIALECTS_MODELICA_OPINTERFACES_H
