#ifndef MARCO_DIALECTS_MODELICA_MODELICATYPES_H
#define MARCO_DIALECTS_MODELICA_MODELICATYPES_H

#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "marco/dialects/modelica/ModelicaTypes.h.inc"

namespace mlir::modelica
{
  //mlir::Type parseModelicaType(mlir::DialectAsmParser& parser);

  //void printModelicaType(mlir::Type type, mlir::DialectAsmPrinter& printer);
}

#endif // MARCO_DIALECTS_MODELICA_MODELICATYPES_H
