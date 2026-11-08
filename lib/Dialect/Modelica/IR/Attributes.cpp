#include "marco/Dialect/Modelica/IR/Attributes.h"
#include "marco/Dialect/Modelica/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::modelica;

//===----------------------------------------------------------------------===//
// Tablegen attribute definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/Modelica/IR/Attributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

namespace mlir::modelica {
void ModelicaDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "marco/Dialect/Modelica/IR/Attributes.cpp.inc"

      >();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Literal values
//===---------------------------------------------------------------------===//

namespace {
/// Parse the optional unit of a numeric literal.
mlir::ParseResult parseOptionalUnit(mlir::AsmParser &parser,
                                    mlir::StringAttr &unit) {
  if (mlir::failed(parser.parseOptionalKeyword("unit"))) {
    return mlir::success();
  }

  std::string value;

  if (parser.parseString(&value)) {
    return mlir::failure();
  }

  unit = parser.getBuilder().getStringAttr(value);
  return mlir::success();
}

void printOptionalUnit(mlir::AsmPrinter &printer, mlir::StringAttr unit) {
  if (unit) {
    printer << " unit ";
    printer.printString(unit.getValue());
  }
}

// The parsing functions below parse the syntax of a specific kind of literal,
// and verify the resulting attribute against the given type.

mlir::Attribute parseBooleanLiteral(mlir::AsmParser &parser, mlir::Type type) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  bool value = false;

  if (mlir::succeeded(parser.parseOptionalKeyword("true"))) {
    value = true;
  } else if (parser.parseKeyword("false")) {
    return {};
  }

  return BooleanAttr::getChecked([&]() { return parser.emitError(loc); },
                                 parser.getContext(), type, value);
}

mlir::Attribute parseIntegerLiteral(mlir::AsmParser &parser, mlir::Type type) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  int64_t value;
  mlir::StringAttr unit;

  if (parser.parseInteger(value) || parseOptionalUnit(parser, unit)) {
    return {};
  }

  return IntegerAttr::getChecked([&]() { return parser.emitError(loc); },
                                 parser.getContext(), type, value, unit);
}

mlir::Attribute parseRealLiteral(mlir::AsmParser &parser, mlir::Type type) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  double value;
  mlir::StringAttr unit;

  if (parser.parseFloat(value) || parseOptionalUnit(parser, unit)) {
    return {};
  }

  return RealAttr::getChecked([&]() { return parser.emitError(loc); },
                              parser.getContext(), type, llvm::APFloat(value),
                              unit);
}

mlir::Attribute parseStringLiteral(mlir::AsmParser &parser, mlir::Type type) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  std::string value;

  if (parser.parseString(&value)) {
    return {};
  }

  return StringValueAttr::getChecked([&]() { return parser.emitError(loc); },
                                     parser.getContext(), type, value);
}

mlir::Attribute parseEnumLiteral(mlir::AsmParser &parser, mlir::Type type) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  std::string literal;
  int64_t ordinal;

  if (parser.parseString(&literal) || parser.parseEqual() ||
      parser.parseInteger(ordinal)) {
    return {};
  }

  return EnumLiteralAttr::getChecked([&]() { return parser.emitError(loc); },
                                     parser.getContext(), type, literal,
                                     ordinal);
}

/// Parse the value of a scalar literal whose kind is determined by the given
/// type, as done for the elements of constant arrays.
mlir::Attribute parseScalarValue(mlir::AsmParser &parser, mlir::Type type) {
  if (mlir::isa<BooleanType>(type)) {
    return parseBooleanLiteral(parser, type);
  }

  if (mlir::isa<IntegerType>(type)) {
    return parseIntegerLiteral(parser, type);
  }

  if (mlir::isa<RealType>(type)) {
    return parseRealLiteral(parser, type);
  }

  if (mlir::isa<StringType>(type)) {
    return parseStringLiteral(parser, type);
  }

  if (mlir::isa<EnumerationType>(type)) {
    return parseEnumLiteral(parser, type);
  }

  parser.emitError(parser.getCurrentLocation(), "unsupported literal type ")
      << type;

  return {};
}

/// Print the value of a scalar literal, without the attribute mnemonic.
void printScalarValue(mlir::AsmPrinter &printer, mlir::Attribute attr) {
  llvm::TypeSwitch<mlir::Attribute>(attr)
      .Case([&](BooleanAttr literal) {
        printer << (literal.getValue() ? "true" : "false");
      })
      .Case([&](IntegerAttr literal) {
        printer << literal.getValue();
        printOptionalUnit(printer, literal.getUnit());
      })
      .Case([&](RealAttr literal) {
        printer.printFloat(literal.getValue());
        printOptionalUnit(printer, literal.getUnit());
      })
      .Case([&](StringValueAttr literal) {
        printer.printString(literal.getValue());
      })
      .Case([&](EnumLiteralAttr literal) {
        printer.printString(literal.getLiteral());
        printer << " = " << literal.getOrdinal();
      })
      .Default([&](mlir::Attribute other) { printer << other; });
}

/// Verify that the type of a literal is the expected one.
template <typename ExpectedType>
mlir::LogicalResult
verifyLiteralType(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  mlir::Type type) {
  if (!mlir::isa<ExpectedType>(type)) {
    return emitError() << "invalid type " << type << " for the literal";
  }

  return mlir::success();
}

mlir::Type getDefaultType(mlir::MLIRContext *context, mlir::Type type,
                          mlir::Type defaultType) {
  return type ? type : defaultType;
}
} // namespace

//===---------------------------------------------------------------------===//
// BooleanAttr
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::Attribute BooleanAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  type = getDefaultType(parser.getContext(), type,
                        BooleanType::get(parser.getContext()));

  return parseBooleanLiteral(parser, type);
}

void BooleanAttr::print(mlir::AsmPrinter &printer) const {
  printer << " ";
  printScalarValue(printer, *this);
}

mlir::LogicalResult
BooleanAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                    mlir::Type type, bool value) {
  return verifyLiteralType<BooleanType>(emitError, type);
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// IntegerAttr
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::Attribute IntegerAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  type = getDefaultType(parser.getContext(), type,
                        IntegerType::get(parser.getContext()));

  return parseIntegerLiteral(parser, type);
}

void IntegerAttr::print(mlir::AsmPrinter &printer) const {
  printer << " ";
  printScalarValue(printer, *this);
}

mlir::LogicalResult
IntegerAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                    mlir::Type type, int64_t value, mlir::StringAttr unit) {
  return verifyLiteralType<IntegerType>(emitError, type);
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// RealAttr
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::Attribute RealAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  type = getDefaultType(parser.getContext(), type,
                        RealType::get(parser.getContext()));

  return parseRealLiteral(parser, type);
}

void RealAttr::print(mlir::AsmPrinter &printer) const {
  printer << " ";
  printScalarValue(printer, *this);
}

mlir::LogicalResult
RealAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                 mlir::Type type, llvm::APFloat value, mlir::StringAttr unit) {
  return verifyLiteralType<RealType>(emitError, type);
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// StringValueAttr
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::Attribute StringValueAttr::parse(mlir::AsmParser &parser,
                                       mlir::Type type) {
  type = getDefaultType(parser.getContext(), type,
                        StringType::get(parser.getContext()));

  return parseStringLiteral(parser, type);
}

void StringValueAttr::print(mlir::AsmPrinter &printer) const {
  printer << " ";
  printScalarValue(printer, *this);
}

mlir::LogicalResult StringValueAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type type,
    llvm::StringRef value) {
  return verifyLiteralType<StringType>(emitError, type);
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// EnumLiteralAttr
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::Attribute EnumLiteralAttr::parse(mlir::AsmParser &parser,
                                       mlir::Type type) {
  if (!type) {
    parser.emitError(parser.getCurrentLocation(),
                     "enumeration literals require an explicit type");

    return {};
  }

  return parseEnumLiteral(parser, type);
}

void EnumLiteralAttr::print(mlir::AsmPrinter &printer) const {
  printer << " ";
  printScalarValue(printer, *this);
}

mlir::LogicalResult EnumLiteralAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type type,
    llvm::StringRef literal, int64_t ordinal) {
  if (mlir::failed(verifyLiteralType<EnumerationType>(emitError, type))) {
    return mlir::failure();
  }

  if (ordinal < 1) {
    return emitError() << "the ordinal value of an enumeration literal must "
                          "be greater than zero";
  }

  return mlir::success();
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// ArrayValueAttr
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::Attribute ArrayValueAttr::parse(mlir::AsmParser &parser,
                                      mlir::Type type) {
  auto tensorType = mlir::dyn_cast_or_null<mlir::RankedTensorType>(type);

  if (!tensorType) {
    parser.emitError(parser.getCurrentLocation(),
                     "constant arrays require an explicit tensor type");

    return {};
  }

  llvm::SmallVector<mlir::Attribute> elements;

  auto parseElement = [&]() -> mlir::ParseResult {
    mlir::Attribute element =
        parseScalarValue(parser, tensorType.getElementType());

    if (!element) {
      return mlir::failure();
    }

    elements.push_back(element);
    return mlir::success();
  };

  if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Square,
                                     parseElement)) {
    return {};
  }

  return getChecked(
      [&]() { return parser.emitError(parser.getCurrentLocation()); },
      parser.getContext(), type, elements);
}

void ArrayValueAttr::print(mlir::AsmPrinter &printer) const {
  printer << " [";

  llvm::interleaveComma(getElements(), printer, [&](mlir::Attribute element) {
    printScalarValue(printer, element);
  });

  printer << "]";
}

mlir::LogicalResult ArrayValueAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type type,
    llvm::ArrayRef<mlir::Attribute> elements) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);

  if (!tensorType || !tensorType.hasStaticShape()) {
    return emitError() << "constant arrays must have a statically shaped "
                          "tensor type, but got "
                       << type;
  }

  if (static_cast<int64_t>(elements.size()) != tensorType.getNumElements()) {
    return emitError() << "expected " << tensorType.getNumElements()
                       << " elements, but got " << elements.size();
  }

  for (mlir::Attribute element : elements) {
    auto typedElement = mlir::dyn_cast<mlir::TypedAttr>(element);

    if (!typedElement ||
        typedElement.getType() != tensorType.getElementType() ||
        mlir::isa<ArrayValueAttr>(element)) {
      return emitError() << "invalid element " << element
                         << " for constant array of type " << type;
    }
  }

  return mlir::success();
}

mlir::RankedTensorType ArrayValueAttr::getTensorType() const {
  return mlir::cast<mlir::RankedTensorType>(getType());
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// DimensionAttr
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
mlir::Attribute DimensionAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  mlir::MLIRContext *context = parser.getContext();

  if (mlir::succeeded(parser.parseOptionalColon())) {
    return DimensionAttr::get(context, DimensionKind::colon);
  }

  if (mlir::succeeded(parser.parseOptionalQuestion())) {
    return DimensionAttr::get(context, DimensionKind::expression);
  }

  int64_t size;

  if (mlir::OptionalParseResult parseResult = parser.parseOptionalInteger(size);
      parseResult.has_value()) {
    if (mlir::failed(*parseResult)) {
      return {};
    }

    return DimensionAttr::getChecked(
        [&]() { return parser.emitError(parser.getCurrentLocation()); },
        context, DimensionKind::fixed, size, mlir::SymbolRefAttr());
  }

  if (mlir::succeeded(parser.parseOptionalKeyword("Boolean"))) {
    return DimensionAttr::get(context, DimensionKind::boolean);
  }

  mlir::SymbolRefAttr enumeration;

  if (parser.parseAttribute(enumeration)) {
    return {};
  }

  return DimensionAttr::get(context, enumeration);
}

void DimensionAttr::print(mlir::AsmPrinter &printer) const {
  printer << " ";

  switch (getKind()) {
  case DimensionKind::colon:
    printer << ":";
    break;

  case DimensionKind::fixed:
    printer << getSize();
    break;

  case DimensionKind::expression:
    printer << "?";
    break;

  case DimensionKind::boolean:
    printer << "Boolean";
    break;

  case DimensionKind::enumeration:
    printer.printAttribute(getEnumeration());
    break;
  }
}

mlir::LogicalResult
DimensionAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                      DimensionKind kind, int64_t size,
                      mlir::SymbolRefAttr enumeration) {
  if ((kind == DimensionKind::enumeration) != static_cast<bool>(enumeration)) {
    return emitError() << "an enumeration type must be specified if and only "
                          "if the dimension is indexed by an enumeration";
  }

  if (kind == DimensionKind::fixed && size < 0) {
    return emitError() << "the size of a dimension can't be negative";
  }

  if (kind != DimensionKind::fixed && size != -1) {
    return emitError() << "a size can be specified only for fixed dimensions";
  }

  return mlir::success();
}
} // namespace mlir::modelica
