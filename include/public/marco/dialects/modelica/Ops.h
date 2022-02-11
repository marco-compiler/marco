#ifndef MARCO_DIALECTS_MODELICA_MODELICAOPS_H
#define MARCO_DIALECTS_MODELICA_MODELICAOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "marco/dialects/modelica/Modelica.h.inc"

#endif // MARCO_DIALECTS_MODELICA_MODELICAOPS_H
