#include "marco/Dialect/Modelica/IR/Attributes.h"
#include "marco/Dialect/Modelica/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::modelica;

//===----------------------------------------------------------------------===//
// Custom parameters and directives
//===----------------------------------------------------------------------===//

namespace {
mlir::FailureOr<bool> parseBooleanValue(mlir::AsmParser &parser) {
  if (mlir::succeeded(parser.parseOptionalKeyword("true"))) {
    return true;
  }

  if (parser.parseKeyword("false")) {
    return mlir::failure();
  }

  return false;
}

void printBooleanValue(mlir::AsmPrinter &printer, bool value) {
  printer << (value ? "true" : "false");
}

mlir::FailureOr<llvm::APFloat> parseRealValue(mlir::AsmParser &parser) {
  double value;

  if (parser.parseFloat(value)) {
    return mlir::failure();
  }

  return llvm::APFloat(value);
}

/// Parse a scalar literal of the given type, using the same syntax of the
/// attribute representing it.
mlir::Attribute parseScalarValue(mlir::AsmParser &parser, mlir::Type type);

/// Print a scalar literal without its type.
void printScalarValue(mlir::AsmPrinter &printer, mlir::Attribute attr);

mlir::ParseResult
parseArrayElements(mlir::AsmParser &parser,
                   llvm::SmallVectorImpl<mlir::Attribute> &elements,
                   mlir::Type type) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);

  if (!tensorType) {
    return parser.emitError(parser.getCurrentLocation(),
                            "constant arrays require a tensor type");
  }

  return parser.parseCommaSeparatedList(
      mlir::AsmParser::Delimiter::Square, [&]() -> mlir::ParseResult {
        mlir::Attribute element =
            parseScalarValue(parser, tensorType.getElementType());

        if (!element) {
          return mlir::failure();
        }

        elements.push_back(element);
        return mlir::success();
      });
}

void printArrayElements(mlir::AsmPrinter &printer,
                        llvm::ArrayRef<mlir::Attribute> elements,
                        mlir::Type type) {
  printer << "[";

  llvm::interleaveComma(elements, printer, [&](mlir::Attribute element) {
    printScalarValue(printer, element);
  });

  printer << "]";
}

mlir::ParseResult parseDimension(mlir::AsmParser &parser,
                                 mlir::modelica::DimensionKind &kind,
                                 int64_t &size,
                                 mlir::SymbolRefAttr &enumeration) {
  using mlir::modelica::DimensionKind;
  size = -1;

  if (mlir::succeeded(parser.parseOptionalColon())) {
    kind = DimensionKind::colon;
    return mlir::success();
  }

  if (mlir::succeeded(parser.parseOptionalQuestion())) {
    kind = DimensionKind::expression;
    return mlir::success();
  }

  if (mlir::OptionalParseResult parseResult = parser.parseOptionalInteger(size);
      parseResult.has_value()) {
    kind = DimensionKind::fixed;
    return *parseResult;
  }

  if (mlir::succeeded(parser.parseOptionalKeyword("Boolean"))) {
    kind = DimensionKind::boolean;
    return mlir::success();
  }

  kind = DimensionKind::enumeration;
  return parser.parseAttribute(enumeration);
}

void printDimension(mlir::AsmPrinter &printer,
                    mlir::modelica::DimensionKind kind, int64_t size,
                    mlir::SymbolRefAttr enumeration) {
  using mlir::modelica::DimensionKind;

  switch (kind) {
  case DimensionKind::colon:
    printer << ":";
    break;

  case DimensionKind::fixed:
    printer << size;
    break;

  case DimensionKind::expression:
    printer << "?";
    break;

  case DimensionKind::boolean:
    printer << "Boolean";
    break;

  case DimensionKind::enumeration:
    printer.printAttribute(enumeration);
    break;
  }
}
} // namespace

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
mlir::Attribute parseScalarValue(mlir::AsmParser &parser, mlir::Type type) {
  using namespace mlir::modelica;

  if (mlir::isa<BooleanType>(type)) {
    return BooleanAttr::parse(parser, type);
  }

  if (mlir::isa<IntegerType>(type)) {
    return IntegerAttr::parse(parser, type);
  }

  if (mlir::isa<RealType>(type)) {
    return RealAttr::parse(parser, type);
  }

  if (mlir::isa<StringType>(type)) {
    return StringValueAttr::parse(parser, type);
  }

  if (mlir::isa<EnumerationType>(type)) {
    return EnumLiteralAttr::parse(parser, type);
  }

  parser.emitError(parser.getCurrentLocation(), "unsupported literal type ")
      << type;

  return {};
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

void printOptionalUnit(mlir::AsmPrinter &printer, mlir::StringAttr unit) {
  if (unit) {
    printer << " unit ";
    printer.printString(unit.getValue());
  }
}

void printScalarValue(mlir::AsmPrinter &printer, mlir::Attribute attr) {
  using namespace mlir::modelica;

  llvm::TypeSwitch<mlir::Attribute>(attr)
      .Case([&](BooleanAttr literal) {
        printBooleanValue(printer, literal.getValue());
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
} // namespace

//===---------------------------------------------------------------------===//
// BooleanAttr
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
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
