#ifndef MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_H
#define MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_H

#include "mlir/IR/Attributes.h"

#define GET_ATTRDEF_CLASSES
#include "marco/dialects/modelica/ModelicaAttributes.h.inc"

namespace mlir::modelica
{
  //mlir::Attribute parseModelicaAttribute(mlir::DialectAsmParser& parser);

  //void printModelicaAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& printer);
}

#endif // MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_H
