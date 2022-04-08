#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "marco/Dialect/Modelica/Attributes.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Modelica/Ops.h"
#include "marco/Dialect/Modelica/Types.h"

using namespace ::mlir;
using namespace ::mlir::modelica;

#include "marco/Dialect/Modelica/ModelicaDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Modelica dialect
//===----------------------------------------------------------------------===//

void ModelicaDialect::initialize() {
  addOperations<
      #define GET_OP_LIST
      #include "marco/Dialect/Modelica/Modelica.cpp.inc"
  >();

  addTypes<
      #define GET_TYPEDEF_LIST
      #include "marco/Dialect/Modelica/ModelicaTypes.cpp.inc"
  >();

  addAttributes<
      #define GET_ATTRDEF_LIST
      #include "marco/Dialect/Modelica/ModelicaAttributes.cpp.inc"
  >();
}

Operation* ModelicaDialect::materializeConstant(
    mlir::OpBuilder& builder, mlir::Attribute value, mlir::Type type, mlir::Location loc)
{
  return builder.create<ConstantOp>(loc, type, value);
}

//===----------------------------------------------------------------------===//
// Tablegen type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/Modelica/ModelicaTypes.cpp.inc"

mlir::Type ModelicaDialect::parseType(mlir::DialectAsmParser& parser) const
{
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  llvm::StringRef mnemonic;

  if (parser.parseKeyword(&mnemonic)) {
    return mlir::Type();
  }

  mlir::Type genType;
  mlir::OptionalParseResult parseResult = generatedTypeParser(getContext(), parser, mnemonic, genType);

  if (parseResult.hasValue()) {
    return genType;
  }

  parser.emitError(typeLoc, "Unknown type in 'Modelica' dialect");
  return mlir::Type();
}

void ModelicaDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& os) const
{
  if (mlir::failed(generatedTypePrinter(type, os))) {
    llvm_unreachable("Unexpected 'Modelica' type kind");
  }
}

//===----------------------------------------------------------------------===//
// Tablegen attribute definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/Modelica/ModelicaAttributes.cpp.inc"

mlir::Attribute ModelicaDialect::parseAttribute(DialectAsmParser& parser, mlir::Type type) const
{
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  llvm::StringRef mnemonic;

  if (parser.parseKeyword(&mnemonic)) {
    return mlir::Attribute();
  }

  mlir::Attribute genAttr;
  mlir::OptionalParseResult parseResult = generatedAttributeParser(getContext(), parser, mnemonic, type, genAttr);

  if (parseResult.hasValue()) {
    return genAttr;
  }

  parser.emitError(typeLoc, "Unknown attribute in 'Modelica' dialect");
  return mlir::Attribute();
}

void ModelicaDialect::printAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& os) const
{
  if (mlir::failed(generatedAttributePrinter(attribute, os))) {
    llvm_unreachable("Unexpected 'Modelica' attribute kind");
  }
}
