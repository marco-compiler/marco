#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "marco/Dialect/IDA/Attributes.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Dialect/IDA/Ops.h"
#include "marco/Dialect/IDA/Types.h"

using namespace ::mlir;
using namespace ::mlir::ida;

#include "marco/Dialect/IDA/IDADialect.cpp.inc"

//===----------------------------------------------------------------------===//
// IDA dialect
//===----------------------------------------------------------------------===//

void IDADialect::initialize() {
  addOperations<
      #define GET_OP_LIST
      #include "marco/Dialect/IDA/IDA.cpp.inc"
  >();

  addTypes<
      #define GET_TYPEDEF_LIST
      #include "marco/Dialect/IDA/IDATypes.cpp.inc"
  >();

  addAttributes<
      #define GET_ATTRDEF_LIST
      #include "marco/Dialect/IDA/IDAAttributes.cpp.inc"
  >();
}

//===----------------------------------------------------------------------===//
// Tablegen type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/IDA/IDATypes.cpp.inc"

mlir::Type IDADialect::parseType(mlir::DialectAsmParser& parser) const
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

  parser.emitError(typeLoc, "Unknown type in 'IDA' dialect");
  return mlir::Type();
}

void IDADialect::printType(mlir::Type type, mlir::DialectAsmPrinter& os) const
{
  if (mlir::failed(generatedTypePrinter(type, os))) {
    llvm_unreachable("Unexpected 'IDA' type kind");
  }
}

//===----------------------------------------------------------------------===//
// Tablegen attribute definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/IDA/IDAAttributes.cpp.inc"

/*
mlir::Attribute IDADialect::parseAttribute(DialectAsmParser& parser, mlir::Type type) const
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

  parser.emitError(typeLoc, "Unknown attribute in 'IDA' dialect");
  return mlir::Attribute();
}

void IDADialect::printAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& os) const
{
  if (mlir::failed(generatedAttributePrinter(attribute, os))) {
    llvm_unreachable("Unexpected 'IDA' attribute kind");
  }
}
*/