#ifndef MARCO_DIALECTS_MODELICA_MODELICATYPES_H
#define MARCO_DIALECTS_MODELICA_MODELICATYPES_H

#include "mlir/IR/Types.h"

namespace mlir::modelica
{
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
}

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/Modelica/ModelicaTypes.h.inc"

#endif // MARCO_DIALECTS_MODELICA_MODELICATYPES_H
