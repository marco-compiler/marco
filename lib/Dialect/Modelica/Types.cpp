#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Modelica/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::modelica;
using namespace ::mlir::modelica::detail;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/Modelica/ModelicaTypes.cpp.inc"

//===---------------------------------------------------------------------===//
// ModelicaDialect
//===---------------------------------------------------------------------===//

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

    mlir::OptionalParseResult parseResult =
        generatedTypeParser(parser, &typeTag, genType);

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

      mlir::Type elementType;
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

    if (typeTag == "variable") {
      if (parser.parseLess()) {
        return mlir::Type();
      }

      llvm::SmallVector<int64_t, 3> dimensions;

      if (parser.parseDimensionList(dimensions)) {
        return mlir::Type();
      }

      mlir::Type elementType;

      if (parser.parseType(elementType)) {
        return mlir::Type();
      }

      VariabilityProperty variabilityProperty = VariabilityProperty::none;
      IOProperty ioProperty = IOProperty::none;

      while (mlir::succeeded(parser.parseOptionalComma())) {
        if (mlir::succeeded(parser.parseOptionalKeyword("discrete"))) {
          variabilityProperty = VariabilityProperty::discrete;
        } else if (mlir::succeeded(parser.parseOptionalKeyword("parameter"))) {
          variabilityProperty = VariabilityProperty::parameter;
        } else if (mlir::succeeded(parser.parseOptionalKeyword("constant"))) {
          variabilityProperty = VariabilityProperty::constant;
        } else if (mlir::succeeded(parser.parseOptionalKeyword("input"))) {
          ioProperty = IOProperty::input;
        } else if (mlir::succeeded(parser.parseOptionalKeyword("output"))) {
          ioProperty = IOProperty::output;
        }
      }

      if (parser.parseGreater()) {
        return mlir::Type();
      }

      return VariableType::get(
          dimensions, elementType, variabilityProperty, ioProperty);
    }

    llvm_unreachable("Unexpected 'Modelica' type kind");
    return mlir::Type();
  }

  void ModelicaDialect::printType(
      mlir::Type type, mlir::DialectAsmPrinter& printer) const
  {
    if (mlir::succeeded(generatedTypePrinter(type, printer))) {
      return;
    }

    if (auto arrayType = type.dyn_cast<ArrayType>()) {
      printer << "array<";

      for (int64_t dimension : arrayType.getShape()) {
        if (dimension == ArrayType::kDynamicSize) {
          printer << "?";
        } else {
          printer << dimension;
        }

        printer << "x";
      }

      printer << arrayType.getElementType() << ">";
      return;
    }

    if (auto unrankedArrayType = type.dyn_cast<UnrankedArrayType>()) {
      printer << "array<*x" << unrankedArrayType.getElementType() << ">";
      return;
    }

    if (auto variableType = type.dyn_cast<VariableType>()) {
      printer << "variable<";

      for (int64_t dimension : variableType.getShape()) {
        if (dimension == VariableType::kDynamicSize) {
          printer << "?";
        } else {
          printer << dimension;
        }

        printer << "x";
      }

      printer << variableType.getElementType();

      if (variableType.isDiscrete()) {
        printer << ", discrete";
      } else if (variableType.isParameter()) {
        printer << ", parameter";
      } else if (variableType.isConstant()) {
        printer << ", constant";
      }

      if (variableType.isInput()) {
        printer << ", input";
      } else if (variableType.isOutput()) {
        printer << ", output";
      }

      printer << ">";
      return;
    }

    llvm_unreachable("Unexpected 'Modelica' type kind");
  }

  //===-------------------------------------------------------------------===//
  // BaseArrayType
  //===-------------------------------------------------------------------===//

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
    return type.isIndex() ||
        type.isa<BooleanType, IntegerType, RealType, RecordType>();
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

  BaseArrayType BaseArrayType::cloneWith(
      llvm::Optional<llvm::ArrayRef<int64_t>> shape,
      mlir::Type elementType) const
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

  //===-------------------------------------------------------------------===//
  // ArrayType
  //===-------------------------------------------------------------------===//

  ArrayType ArrayType::get(
      llvm::ArrayRef<int64_t> shape,
      mlir::Type elementType,
      mlir::Attribute memorySpace)
  {
    // Drop default memory space value and replace it with empty attribute.
    memorySpace = skipDefaultMemorySpace(memorySpace);

    return Base::get(
        elementType.getContext(),
        shape,
        elementType,
        memorySpace);
  }

  ArrayType ArrayType::getChecked(
      llvm::function_ref<mlir::InFlightDiagnostic()> emitErrorFn,
      llvm::ArrayRef<int64_t> shape,
      mlir::Type elementType,
      mlir::Attribute memorySpace)
  {
    // Drop default memory space value and replace it with empty attribute.
    memorySpace = skipDefaultMemorySpace(memorySpace);

    return Base::getChecked(
        emitErrorFn,
        elementType.getContext(),
        shape,
        elementType,
        memorySpace);
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
    for (int64_t size : shape) {
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

  //===-------------------------------------------------------------------===//
  // UnrankedArrayType
  //===-------------------------------------------------------------------===//

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
  // VariableType
  //===-------------------------------------------------------------------===//

  VariableType VariableType::get(
      llvm::ArrayRef<int64_t> shape,
      mlir::Type elementType,
      VariabilityProperty variabilityProperty,
      IOProperty ioProperty,
      mlir::Attribute memorySpace)
  {
    // Drop default memory space value and replace it with empty attribute.
    memorySpace = skipDefaultMemorySpace(memorySpace);

    return Base::get(
        elementType.getContext(),
        shape,
        elementType,
        variabilityProperty,
        ioProperty,
        memorySpace);
  }

  VariableType VariableType::getChecked(
      llvm::function_ref<mlir::InFlightDiagnostic()> emitErrorFn,
      llvm::ArrayRef<int64_t> shape,
      mlir::Type elementType,
      VariabilityProperty variabilityProperty,
      IOProperty ioProperty,
      mlir::Attribute memorySpace)
  {
    // Drop default memory space value and replace it with empty attribute.
    memorySpace = skipDefaultMemorySpace(memorySpace);

    return Base::getChecked(
        emitErrorFn,
        elementType.getContext(),
        shape,
        elementType,
        variabilityProperty,
        ioProperty,
        memorySpace);
  }

  mlir::LogicalResult VariableType::verify(
      llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
      llvm::ArrayRef<int64_t> shape,
      mlir::Type elementType,
      VariabilityProperty variabilityProperty,
      IOProperty ioProperty,
      mlir::Attribute memorySpace)
  {
    if (!isValidElementType(elementType)) {
      return emitError() << "invalid variable element type";
    }

    // Negative sizes are not allowed except for `-1` that means dynamic size.
    for (int64_t size : shape) {
      if (size < 0 && size != VariableType::kDynamicSize) {
        return emitError() << "invalid variable size";
      }
    }

    if (!isSupportedMemorySpace(memorySpace)) {
      return emitError() << "unsupported memory space Attribute";
    }

    return mlir::success();
  }

  bool VariableType::hasRank() const
  {
    return true;
  }

  mlir::ShapedType VariableType::cloneWith(
      llvm::Optional<llvm::ArrayRef<int64_t>> shape,
      mlir::Type elementType) const
  {
    VariableType::Builder builder(*shape, elementType);
    builder.setVariabilityProperty(getVariabilityProperty());
    builder.setVisibilityProperty(getVisibilityProperty());
    builder.setMemorySpace(getMemorySpace());
    return builder;
  }

  bool VariableType::isValidElementType(mlir::Type type)
  {
    return type.isIndex() ||
        type.isa<BooleanType, IntegerType, RealType, RecordType>();
  }

  VariableType VariableType::wrap(
      mlir::Type type,
      VariabilityProperty variabilityProperty,
      IOProperty ioProperty)
  {
    if (auto arrayType = type.dyn_cast<ArrayType>()) {
      return VariableType::get(
          arrayType.getShape(),
          arrayType.getElementType(),
          variabilityProperty,
          ioProperty,
          arrayType.getMemorySpace());
    }

    return VariableType::get(llvm::None, type, variabilityProperty, ioProperty);
  }

  ArrayType VariableType::toArrayType() const
  {
    return ArrayType::get(getShape(), getElementType(), getMemorySpace());
  }

  mlir::Type VariableType::unwrap() const
  {
    if (!isScalar()) {
      return toArrayType();
    }

    return getElementType();
  }

  VariableType VariableType::withShape(llvm::ArrayRef<int64_t> shape) const
  {
    return VariableType::get(
        shape,
        getElementType(),
        getVariabilityProperty(),
        getVisibilityProperty());
  }

  VariableType VariableType::withType(mlir::Type type) const
  {
    return VariableType::get(
        getShape(),
        type,
        getVariabilityProperty(),
        getVisibilityProperty());
  }

  VariableType VariableType::withVariabilityProperty(
      VariabilityProperty variabilityProperty) const
  {
    return VariableType::get(
        getShape(),
        getElementType(),
        variabilityProperty,
        getVisibilityProperty());
  }

  VariableType VariableType::withoutVariabilityProperty() const
  {
    return withVariabilityProperty(VariabilityProperty::none);
  }

  VariableType VariableType::asDiscrete() const
  {
    return withVariabilityProperty(VariabilityProperty::discrete);
  }

  VariableType VariableType::asParameter() const
  {
    return withVariabilityProperty(VariabilityProperty::parameter);
  }

  VariableType VariableType::asConstant() const
  {
    return withVariabilityProperty(VariabilityProperty::constant);
  }

  VariableType VariableType::withIOProperty(IOProperty ioProperty) const
  {
    return VariableType::get(
        getShape(),
        getElementType(),
        getVariabilityProperty(),
        ioProperty);
  }

  VariableType VariableType::withoutIOProperty() const
  {
    return withIOProperty(IOProperty::none);
  }

  VariableType VariableType::asInput() const
  {
    return withIOProperty(IOProperty::input);
  }

  VariableType VariableType::asOutput() const
  {
    return withIOProperty(IOProperty::output);
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
    if (memorySpace.isa<
        mlir::IntegerAttr, mlir::StringAttr, mlir::DictionaryAttr>()) {
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
    mlir::IntegerAttr intMemorySpace =
        memorySpace.dyn_cast_or_null<mlir::IntegerAttr>();

    if (intMemorySpace && intMemorySpace.getValue() == 0) {
      return nullptr;
    }

    return memorySpace;
  }
}
