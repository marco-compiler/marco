#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "marco/Dialect/KINSOL/Attributes.h"
#include "marco/Dialect/KINSOL/KINSOLDialect.h"
#include "marco/Dialect/KINSOL/Ops.h"
#include "marco/Dialect/KINSOL/Types.h"

using namespace ::mlir;
using namespace ::mlir::kinsol;

#include "marco/Dialect/KINSOL/KINSOLDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// KINSOL dialect
//===----------------------------------------------------------------------===//

void KINSOLDialect::initialize() {
  addOperations<
      #define GET_OP_LIST
      #include "marco/Dialect/KINSOL/KINSOL.cpp.inc"
  >();

  addTypes<
      #define GET_TYPEDEF_LIST
      #include "marco/Dialect/KINSOL/KINSOLTypes.cpp.inc"
  >();

  addAttributes<
      #define GET_ATTRDEF_LIST
      #include "marco/Dialect/KINSOL/KINSOLAttributes.cpp.inc"
  >();
}

//===----------------------------------------------------------------------===//
// Tablegen type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/KINSOL/KINSOLTypes.cpp.inc"

/*
mlir::Type KINSOLDialect::parseType(mlir::DialectAsmParser& parser) const
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

  parser.emitError(typeLoc, "Unknown type in 'KINSOL' dialect");
  return mlir::Type();
}

void KINSOLDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& os) const
{
  if (mlir::failed(generatedTypePrinter(type, os))) {
    llvm_unreachable("Unexpected 'KINSOL' type kind");
  }
}
 */

//===----------------------------------------------------------------------===//
// Tablegen attribute definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/KINSOL/KINSOLAttributes.cpp.inc"

/*
mlir::Attribute KINSOLDialect::parseAttribute(DialectAsmParser& parser, mlir::Type type) const
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

  parser.emitError(typeLoc, "Unknown attribute in 'KINSOL' dialect");
  return mlir::Attribute();
}

void KINSOLDialect::printAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& os) const
{
  if (mlir::failed(generatedAttributePrinter(attribute, os))) {
    llvm_unreachable("Unexpected 'KINSOL' attribute kind");
  }
}
*/
