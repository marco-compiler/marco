#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "marco/dialects/modelica/ModelicaAttributes.h"
#include "marco/dialects/modelica/ModelicaDialect.h"
#include "marco/dialects/modelica/ModelicaOps.h"
#include "marco/dialects/modelica/ModelicaTypes.h"

using namespace ::mlir;
using namespace ::mlir::modelica;

#include "marco/dialects/modelica/ModelicaDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Modelica dialect
//===----------------------------------------------------------------------===//

void ModelicaDialect::initialize() {
  addOperations<
      #define GET_OP_LIST
      #include "marco/dialects/modelica/Modelica.cpp.inc"
  >();

  addTypes<
      #define GET_TYPEDEF_LIST
      #include "marco/dialects/modelica/ModelicaTypes.cpp.inc"
  >();

  addAttributes<
      #define GET_ATTRDEF_LIST
      #include "marco/dialects/modelica/ModelicaAttributes.cpp.inc"
  >();
}

mlir::Type ModelicaDialect::parseType(mlir::DialectAsmParser& parser) const
{
  return mlir::Type();
  //return parseModelicaType(parser);
}

void ModelicaDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const
{
  //return printModelicaType(type, printer);
}

mlir::Attribute ModelicaDialect::parseAttribute(DialectAsmParser& parser, Type type) const
{
  return mlir::Attribute();
  //return parseModelicaType(parser);
}

void ModelicaDialect::printAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& printer) const
{
  //return printModelicaType(type, printer);
}

//===----------------------------------------------------------------------===//
// Tablegen type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/dialects/modelica/ModelicaTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Tablegen attribute definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/dialects/modelica/ModelicaAttributes.cpp.inc"
