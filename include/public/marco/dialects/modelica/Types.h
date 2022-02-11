#ifndef MARCO_DIALECTS_MODELICA_MODELICATYPES_H
#define MARCO_DIALECTS_MODELICA_MODELICATYPES_H

#include "mlir/IR/Types.h"

enum class ArrayAllocationScope
{
  unknown,
  stack,
  heap
};

enum class MemberAllocationScope
{
  stack,
  heap
};

#define GET_TYPEDEF_CLASSES
#include "marco/dialects/modelica/ModelicaTypes.h.inc"

#endif // MARCO_DIALECTS_MODELICA_MODELICATYPES_H
