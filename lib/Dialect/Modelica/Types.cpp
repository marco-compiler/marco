#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Modelica/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::modelica;
using namespace ::mlir::modelica::detail;

//===----------------------------------------------------------------------===//
// Tablegen type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/Modelica/ModelicaTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// ModelicaDialect
//===----------------------------------------------------------------------===//

namespace mlir::modelica
{
  void ModelicaDialect::registerTypes()
  {
    addTypes<
      #define GET_TYPEDEF_LIST
      #include "marco/Dialect/Modelica/ModelicaTypes.cpp.inc"
    >();
  }
}

namespace mlir::modelica
{
  mlir::Type ModelicaDialect::parseType(mlir::DialectAsmParser& parser) const
  {
    llvm::StringRef typeTag;
    mlir::Type genType;

    mlir::OptionalParseResult parseResult = generatedTypeParser(parser, &typeTag, genType);

    if (parseResult.has_value()) {
      return genType;
    }

    if (typeTag == "array") {
      bool isUnranked;
      llvm::SmallVector<int64_t, 3> dimensions;

      if (parser.parseLess()) {
        return mlir::Type();
      }

      if (mlir::succeeded(parser.parseOptionalStar())) {
        isUnranked = true;

        if (parser.parseXInDimensionList()) {
          return mlir::Type();
        }
      } else {
        isUnranked = false;

        if (parser.parseDimensionList(dimensions)) {
          return mlir::Type();
        }
      }

      Type elementType;
      mlir::Attribute memorySpace;

      if (parser.parseType(elementType)) {
        return mlir::Type();
      }

      if (mlir::succeeded(parser.parseOptionalComma())) {
        if (parser.parseAttribute(memorySpace)) {
          return mlir::Type();
        }
      }

      if (isUnranked) {
        return UnrankedArrayType::get(elementType, memorySpace);
      }

      return ArrayType::get(dimensions, elementType, memorySpace);
    }

    if (typeTag == "member") {
      if (parser.parseLess()) {
        return mlir::Type();
      }

      llvm::SmallVector<int64_t, 3> dimensions;

      if (parser.parseDimensionList(dimensions)) {
        return mlir::Type();
      }

      Type elementType;

      if (parser.parseType(elementType)) {
        return mlir::Type();
      }

      bool isParameter = false;
      IOProperty ioProperty = IOProperty::none;

      while (mlir::succeeded(parser.parseOptionalComma())) {
        if (mlir::succeeded(parser.parseOptionalKeyword("parameter"))) {
          isParameter = true;
        } else if (mlir::succeeded(parser.parseOptionalKeyword("input"))) {
          ioProperty = IOProperty::input;
        } else if (mlir::succeeded(parser.parseOptionalKeyword("output"))) {
          ioProperty = IOProperty::output;
        }
      }

      if (parser.parseGreater()) {
        return mlir::Type();
      }

      return MemberType::get(dimensions, elementType, isParameter, ioProperty);
    }

    llvm_unreachable("Unexpected 'Modelica' type kind");
    return mlir::Type();
  }

  void ModelicaDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const
  {
    if (mlir::succeeded(generatedTypePrinter(type, printer))) {
      return;
    }

    if (auto arrayType = type.dyn_cast<ArrayType>()) {
      printer << "array<";

      for (const auto& dimension : arrayType.getShape()) {
        printer << (dimension == ArrayType::kDynamicSize ? "?" : std::to_string(dimension)) << "x";
      }

      printer << arrayType.getElementType() << ">";
      return;
    }

    if (auto unrankedArrayType = type.dyn_cast<UnrankedArrayType>()) {
      printer << "array<*x" << unrankedArrayType.getElementType() << ">";
      return;
    }

    if (auto memberType = type.dyn_cast<MemberType>()) {
      printer << "member<";

      for (const auto& dimension : memberType.getShape()) {
        printer << (dimension == MemberType::kDynamicSize ? "?" : std::to_string(dimension)) << "x";
      }

      printer << memberType.getElementType();

      if (memberType.isParameter()) {
        printer << ", parameter";
      }

      if (memberType.isInput()) {
        printer << ", input";
      } else if (memberType.isOutput()) {
        printer << ", output";
      }

      printer << ">";
      return;
    }

    llvm_unreachable("Unexpected 'Modelica' type kind");
  }

  //===----------------------------------------------------------------------===//
  // BaseArrayType
  //===----------------------------------------------------------------------===//

  bool BaseArrayType::classof(mlir::Type type)
  {
    return type.isa<ArrayType, UnrankedArrayType>();
  }

  BaseArrayType::operator mlir::ShapedType() const
  {
    return cast<mlir::ShapedType>();
  }

  bool BaseArrayType::isValidElementType(mlir::Type type)
  {
    return type.isIndex() || type.isa<BooleanType, IntegerType, RealType>();
  }

  mlir::Type BaseArrayType::getElementType() const
  {
    return llvm::TypeSwitch<BaseArrayType, mlir::Type>(*this)
        .Case<ArrayType, UnrankedArrayType>(
            [](auto type) {
              return type.getElementType();
            });
  }

  bool BaseArrayType::hasRank() const
  {
    return !isa<UnrankedArrayType>();
  }

  llvm::ArrayRef<int64_t> BaseArrayType::getShape() const
  {
    return cast<ArrayType>().getShape();
  }

  mlir::Attribute BaseArrayType::getMemorySpace() const
  {
    if (auto rankedArrayTy = dyn_cast<ArrayType>()) {
      return rankedArrayTy.getMemorySpace();
    }

    return cast<UnrankedArrayType>().getMemorySpace();
  }

  BaseArrayType BaseArrayType::cloneWith(llvm::Optional<llvm::ArrayRef<int64_t>> shape, mlir::Type elementType) const
  {
    if (isa<UnrankedArrayType>()) {
      if (!shape) {
        return UnrankedArrayType::get(elementType, getMemorySpace());
      }

      ArrayType::Builder builder(*shape, elementType);
      builder.setMemorySpace(getMemorySpace());
      return builder;
    }

    ArrayType::Builder builder(cast<ArrayType>());

    if (shape) {
      builder.setShape(*shape);
    }

    builder.setElementType(elementType);
    return builder;
  }

  //===----------------------------------------------------------------------===//
  // ArrayType
  //===----------------------------------------------------------------------===//

  ArrayType ArrayType::get(llvm::ArrayRef<int64_t> shape, mlir::Type elementType, mlir::Attribute memorySpace)
  {
    // Drop default memory space value and replace it with empty attribute.
    memorySpace = skipDefaultMemorySpace(memorySpace);

    return Base::get(elementType.getContext(), shape, elementType, memorySpace);
  }

  ArrayType ArrayType::getChecked(
      llvm::function_ref<mlir::InFlightDiagnostic()> emitErrorFn,
      llvm::ArrayRef<int64_t> shape,
      mlir::Type elementType,
      mlir::Attribute memorySpace)
  {
    // Drop default memory space value and replace it with empty attribute.
    memorySpace = skipDefaultMemorySpace(memorySpace);

    return Base::getChecked(emitErrorFn, elementType.getContext(), shape, elementType, memorySpace);
  }

  mlir::LogicalResult ArrayType::verify(
      llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
      llvm::ArrayRef<int64_t> shape,
      mlir::Type elementType,
      mlir::Attribute memorySpace)
  {
    if (!BaseArrayType::isValidElementType(elementType)) {
      return emitError() << "invalid array element type";
    }

    // Negative sizes are not allowed except for `-1` that means dynamic size.
    for (const auto& size : shape) {
      if (size < 0 && size != ArrayType::kDynamicSize) {
        return emitError() << "invalid array size";
      }
    }

    if (!isSupportedMemorySpace(memorySpace)) {
      return emitError() << "unsupported memory space Attribute";
    }

    return mlir::success();
  }

  bool ArrayType::isScalar() const
  {
    return getRank() == 0;
  }

  ArrayType ArrayType::slice(unsigned int subscriptsAmount) const
  {
    auto shape = getShape();
    assert(subscriptsAmount <= shape.size() && "Too many subscriptions");
    llvm::SmallVector<int64_t, 3> resultShape;

    for (size_t i = subscriptsAmount, e = shape.size(); i < e; ++i) {
      resultShape.push_back(shape[i]);
    }

    return ArrayType::get(resultShape, getElementType());
  }

  ArrayType ArrayType::toElementType(mlir::Type elementType) const
  {
    return ArrayType::get(getShape(), elementType);
  }

  bool ArrayType::canBeOnStack() const
  {
    return hasStaticShape();
  }

  //===----------------------------------------------------------------------===//
  // UnrankedArrayType
  //===----------------------------------------------------------------------===//

  mlir::LogicalResult UnrankedArrayType::verify(
      llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
      mlir::Type elementType,
      mlir::Attribute memorySpace)
  {
    if (!BaseArrayType::isValidElementType(elementType)) {
      return emitError() << "invalid array element type";
    }

    if (!isSupportedMemorySpace(memorySpace)) {
      return emitError() << "unsupported memory space Attribute";
    }

    return mlir::success();
  }

  //===-------------------------------------------------------------------===//
  // MemberType
  //===-------------------------------------------------------------------===//

  MemberType MemberType::get(
      llvm::ArrayRef<int64_t> shape,
      mlir::Type elementType,
      bool isParameter,
      IOProperty ioProperty,
      mlir::Attribute memorySpace)
  {
    // Drop default memory space value and replace it with empty attribute.
    memorySpace = skipDefaultMemorySpace(memorySpace);

    return Base::get(
        elementType.getContext(),
        shape,
        elementType,
        isParameter,
        ioProperty,
        memorySpace);
  }

  MemberType MemberType::getChecked(
      llvm::function_ref<mlir::InFlightDiagnostic()> emitErrorFn,
      llvm::ArrayRef<int64_t> shape,
      mlir::Type elementType,
      bool isParameter,
      IOProperty ioProperty,
      mlir::Attribute memorySpace)
  {
    // Drop default memory space value and replace it with empty attribute.
    memorySpace = skipDefaultMemorySpace(memorySpace);

    return Base::get(
        elementType.getContext(),
        shape,
        elementType,
        isParameter,
        ioProperty,
        memorySpace);
  }

  mlir::LogicalResult MemberType::verify(
      llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
      llvm::ArrayRef<int64_t> shape,
      mlir::Type elementType,
      bool isParameter,
      IOProperty ioProperty,
      mlir::Attribute memorySpace)
  {
    if (!isValidElementType(elementType)) {
      return emitError() << "invalid member element type";
    }

    // Negative sizes are not allowed except for `-1` that means dynamic size.
    for (const auto& size : shape) {
      if (size < 0 && size != MemberType::kDynamicSize) {
        return emitError() << "invalid member size";
      }
    }

    if (!isSupportedMemorySpace(memorySpace)) {
      return emitError() << "unsupported memory space Attribute";
    }

    return mlir::success();
  }

  bool MemberType::hasRank() const
  {
    return !getShape().empty();
  }

  mlir::ShapedType MemberType::cloneWith(
      llvm::Optional<llvm::ArrayRef<int64_t>> shape,
      mlir::Type elementType) const
  {
    MemberType::Builder builder(*shape, elementType);
    builder.setParameterProperty(getParameterProperty());
    builder.setVisibilityProperty(getVisibilityProperty());
    builder.setMemorySpace(getMemorySpace());
    return builder;
  }

  bool MemberType::isValidElementType(mlir::Type type)
  {
    return type.isIndex() || type.isa<BooleanType, IntegerType, RealType>();
  }

  MemberType MemberType::wrap(
      mlir::Type type, bool isParameter, IOProperty ioProperty)
  {
    if (auto arrayType = type.dyn_cast<ArrayType>()) {
      return MemberType::get(
          arrayType.getShape(),
          arrayType.getElementType(),
          isParameter,
          ioProperty,
          arrayType.getMemorySpace());
    }

    return MemberType::get(llvm::None, type, isParameter, ioProperty);
  }

  ArrayType MemberType::toArrayType() const
  {
    return ArrayType::get(getShape(), getElementType(), getMemorySpace());
  }

  mlir::Type MemberType::unwrap() const
  {
    if (hasRank()) {
      return toArrayType();
    }

    return getElementType();
  }

  MemberType MemberType::withShape(llvm::ArrayRef<int64_t> shape) const
  {
    return MemberType::get(
        shape, getElementType(), isParameter(), getVisibilityProperty());
  }

  MemberType MemberType::withType(mlir::Type type) const
  {
    return MemberType::wrap(type, isParameter(), getVisibilityProperty());
  }

  MemberType MemberType::asNonParameter() const
  {
    return MemberType::get(
        getShape(), getElementType(), false, getVisibilityProperty());
  }

  MemberType MemberType::asParameter() const
  {
    return MemberType::get(
        getShape(), getElementType(), true, getVisibilityProperty());
  }

  MemberType MemberType::withIOProperty(IOProperty ioProperty) const
  {
    return MemberType::get(
        getShape(), getElementType(), isParameter(), ioProperty);
  }
}

namespace mlir::modelica::detail
{
  bool isSupportedMemorySpace(mlir::Attribute memorySpace)
  {
    // Empty attribute is allowed as default memory space.
    if (!memorySpace) {
      return true;
    }

    // Supported built-in attributes.
    if (memorySpace.isa<mlir::IntegerAttr, mlir::StringAttr, mlir::DictionaryAttr>()) {
      return true;
    }

    // Allow custom dialect attributes.
    if (!isa<mlir::BuiltinDialect>(memorySpace.getDialect())) {
      return true;
    }

    return false;
  }

  mlir::Attribute skipDefaultMemorySpace(mlir::Attribute memorySpace)
  {
    mlir::IntegerAttr intMemorySpace = memorySpace.dyn_cast_or_null<mlir::IntegerAttr>();

    if (intMemorySpace && intMemorySpace.getValue() == 0) {
      return nullptr;
    }

    return memorySpace;
  }
}
