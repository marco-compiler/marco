#include "marco/Dialect/BaseModelica/IR/Types.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::detail;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/BaseModelica/IR/BaseModelicaTypes.cpp.inc"

//===---------------------------------------------------------------------===//
// ModelicaDialect
//===---------------------------------------------------------------------===//

namespace mlir::bmodelica {
void BaseModelicaDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "marco/Dialect/BaseModelica/IR/BaseModelicaTypes.cpp.inc"
      >();
}
} // namespace mlir::bmodelica

namespace {
template <typename Attr>
Attr getTypeDLDictValue(mlir::DataLayoutEntryInterface entry,
                        llvm::StringRef key) {
  auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(entry.getValue());

  if (!arrayAttr) {
    return {};
  }

  if (arrayAttr.size() != 2) {
    return {};
  }

  auto keyAttr = mlir::dyn_cast<mlir::StringAttr>(arrayAttr[0]);

  if (!keyAttr || keyAttr != key) {
    return {};
  }

  return mlir::dyn_cast<Attr>(arrayAttr[1]);
}

std::optional<uint64_t> getTypeSizeInBits(mlir::DataLayoutEntryListRef params) {
  for (mlir::DataLayoutEntryInterface entry : params) {
    auto valueAttr = getTypeDLDictValue<mlir::IntegerAttr>(entry, "size");

    if (!valueAttr) {
      continue;
    }

    return valueAttr.getValue().getZExtValue();
  }

  return std::nullopt;
}
} // namespace

namespace mlir::bmodelica {
//===---------------------------------------------------------------------===//
// BooleanType
//===---------------------------------------------------------------------===//
llvm::TypeSize
BooleanType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                               mlir::DataLayoutEntryListRef params) const {
  if (auto size = ::getTypeSizeInBits(params)) {
    return llvm::TypeSize::getFixed(*size);
  }

  return llvm::TypeSize::getFixed(1);
}

uint64_t
BooleanType::getABIAlignment(const ::mlir::DataLayout &dataLayout,
                             ::mlir::DataLayoutEntryListRef params) const {
  uint64_t size = getTypeSize(dataLayout, params).getKnownMinValue();

  if (size >= 8) {
    return 4;
  }

  return llvm::bit_ceil(size);
}

uint64_t BooleanType::getPreferredAlignment(
    const ::mlir::DataLayout &dataLayout,
    ::mlir::DataLayoutEntryListRef params) const {
  uint64_t size = getTypeSize(dataLayout, params).getKnownMinValue();

  if (size >= 8) {
    return 4;
  }

  return llvm::bit_ceil(size);
}

//===---------------------------------------------------------------------===//
// IntegerType
//===---------------------------------------------------------------------===//
llvm::TypeSize
IntegerType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                               mlir::DataLayoutEntryListRef params) const {
  if (auto size = ::getTypeSizeInBits(params)) {
    return llvm::TypeSize::getFixed(*size);
  }

  return llvm::TypeSize::getFixed(64);
}

uint64_t
IntegerType::getABIAlignment(const ::mlir::DataLayout &dataLayout,
                             ::mlir::DataLayoutEntryListRef params) const {
  uint64_t size = getTypeSize(dataLayout, params).getKnownMinValue();

  if (size >= 8) {
    return 4;
  }

  return llvm::bit_ceil(size);
}

uint64_t IntegerType::getPreferredAlignment(
    const ::mlir::DataLayout &dataLayout,
    ::mlir::DataLayoutEntryListRef params) const {
  uint64_t size = getTypeSize(dataLayout, params).getKnownMinValue();

  if (size >= 8) {
    return 4;
  }

  return llvm::bit_ceil(size);
}

//===---------------------------------------------------------------------===//
// RealType
//===---------------------------------------------------------------------===//
llvm::TypeSize
RealType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                            mlir::DataLayoutEntryListRef params) const {
  if (auto size = ::getTypeSizeInBits(params)) {
    return llvm::TypeSize::getFixed(*size);
  }

  return llvm::TypeSize::getFixed(64);
}

uint64_t
RealType::getABIAlignment(const ::mlir::DataLayout &dataLayout,
                          ::mlir::DataLayoutEntryListRef params) const {
  return 64;
}

uint64_t
RealType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                ::mlir::DataLayoutEntryListRef params) const {
  return 64;
}

//===-------------------------------------------------------------------===//
// BaseArrayType
//===-------------------------------------------------------------------===//

bool BaseArrayType::classof(mlir::Type type) {
  return type.isa<ArrayType, UnrankedArrayType>();
}

BaseArrayType::operator mlir::ShapedType() const {
  return cast<mlir::ShapedType>();
}

bool BaseArrayType::isValidElementType(mlir::Type type) {
  return type.isIndex() || type.isIntOrFloat() ||
         type.isa<BooleanType, IntegerType, RealType, RecordType>();
}

mlir::Type BaseArrayType::getElementType() const {
  return llvm::TypeSwitch<BaseArrayType, mlir::Type>(*this)
      .Case<ArrayType, UnrankedArrayType>(
          [](auto type) { return type.getElementType(); });
}

bool BaseArrayType::hasRank() const { return !isa<UnrankedArrayType>(); }

llvm::ArrayRef<int64_t> BaseArrayType::getShape() const {
  return cast<ArrayType>().getShape();
}

mlir::Attribute BaseArrayType::getMemorySpace() const {
  if (auto rankedArrayTy = dyn_cast<ArrayType>()) {
    return rankedArrayTy.getMemorySpace();
  }

  return cast<UnrankedArrayType>().getMemorySpace();
}

BaseArrayType
BaseArrayType::cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                         mlir::Type elementType) const {
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

mlir::Type ArrayType::parse(mlir::AsmParser &parser) {
  llvm::SmallVector<int64_t, 3> dimensions;

  mlir::Type elementType;
  mlir::Attribute memorySpace;

  if (parser.parseLess() || parser.parseDimensionList(dimensions) ||
      parser.parseType(elementType)) {
    return {};
  }

  if (mlir::succeeded(parser.parseOptionalComma())) {
    if (parser.parseAttribute(memorySpace)) {
      return {};
    }
  }

  if (parser.parseGreater()) {
    return {};
  }

  return ArrayType::get(dimensions, elementType, memorySpace);
}

void ArrayType::print(mlir::AsmPrinter &printer) const {
  printer << "<";

  for (int64_t dimension : getShape()) {
    if (dimension == ArrayType::kDynamic) {
      printer << "?";
    } else {
      printer << dimension;
    }

    printer << "x";
  }

  printer << getElementType() << ">";
  // TODO print memory space
}

ArrayType ArrayType::get(llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                         mlir::Attribute memorySpace) {
  // Drop default memory space value and replace it with empty attribute.
  memorySpace = skipDefaultMemorySpace(memorySpace);

  return Base::get(elementType.getContext(), shape, elementType, memorySpace);
}

ArrayType ArrayType::getChecked(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitErrorFn,
    llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
    mlir::Attribute memorySpace) {
  // Drop default memory space value and replace it with empty attribute.
  memorySpace = skipDefaultMemorySpace(memorySpace);

  return Base::getChecked(emitErrorFn, elementType.getContext(), shape,
                          elementType, memorySpace);
}

mlir::LogicalResult
ArrayType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                  mlir::Attribute memorySpace) {
  if (!BaseArrayType::isValidElementType(elementType)) {
    return emitError() << "invalid array element type";
  }

  // Negative sizes are not allowed.
  for (int64_t size : shape) {
    if (size < 0 && size != ArrayType::kDynamic) {
      return emitError() << "invalid array size";
    }
  }

  if (!isSupportedMemorySpace(memorySpace)) {
    return emitError() << "unsupported memory space Attribute";
  }

  return mlir::success();
}

bool ArrayType::isScalar() const { return getRank() == 0; }

ArrayType ArrayType::slice(unsigned int subscriptsAmount) const {
  auto shape = getShape();
  assert(subscriptsAmount <= shape.size() && "Too many subscriptions");
  llvm::SmallVector<int64_t, 3> resultShape;

  for (size_t i = subscriptsAmount, e = shape.size(); i < e; ++i) {
    resultShape.push_back(shape[i]);
  }

  return ArrayType::get(resultShape, getElementType());
}

ArrayType ArrayType::toElementType(mlir::Type elementType) const {
  return ArrayType::get(getShape(), elementType);
}

ArrayType ArrayType::withShape(llvm::ArrayRef<int64_t> shape) const {
  return ArrayType::get(shape, getElementType());
}

bool ArrayType::canBeOnStack() const { return hasStaticShape(); }

//===-------------------------------------------------------------------===//
// UnrankedArrayType
//===-------------------------------------------------------------------===//

mlir::Type UnrankedArrayType::parse(mlir::AsmParser &parser) {
  mlir::Type elementType;
  mlir::Attribute memorySpace;

  if (parser.parseLess() || parser.parseType(elementType)) {
    return {};
  }

  if (mlir::succeeded(parser.parseOptionalComma())) {
    if (parser.parseAttribute(memorySpace)) {
      return {};
    }
  }

  if (parser.parseGreater()) {
    return {};
  }

  return UnrankedArrayType::get(elementType, memorySpace);
}

void UnrankedArrayType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getElementType() << ">";
  // TODO print memory space
}

mlir::LogicalResult UnrankedArrayType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type elementType, mlir::Attribute memorySpace) {
  if (!BaseArrayType::isValidElementType(elementType)) {
    return emitError() << "invalid array element type";
  }

  if (!isSupportedMemorySpace(memorySpace)) {
    return emitError() << "unsupported memory space Attribute";
  }

  return mlir::success();
}

//===-------------------------------------------------------------------===//
// RecordType
//===-------------------------------------------------------------------===//

mlir::Operation *
RecordType::getRecordOp(mlir::SymbolTableCollection &symbolTable,
                        mlir::ModuleOp moduleOp) {
  mlir::Operation *result = moduleOp.getOperation();
  result = symbolTable.lookupSymbolIn(result, getName().getRootReference());
  mlir::SymbolRefAttr nameAttr = getName();

  for (mlir::FlatSymbolRefAttr flatSymbolRef : nameAttr.getNestedReferences()) {
    if (result == nullptr) {
      return nullptr;
    }

    result = symbolTable.lookupSymbolIn(result, flatSymbolRef.getAttr());
  }

  return result;
}

//===-------------------------------------------------------------------===//
// VariableType
//===-------------------------------------------------------------------===//

mlir::Type VariableType::parse(mlir::AsmParser &parser) {
  llvm::SmallVector<int64_t, 3> dimensions;
  mlir::Type elementType;

  if (parser.parseLess() || parser.parseDimensionList(dimensions) ||
      parser.parseType(elementType)) {
    return {};
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
    return {};
  }

  return VariableType::get(dimensions, elementType, variabilityProperty,
                           ioProperty);
}

void VariableType::print(mlir::AsmPrinter &printer) const {
  printer << "<";

  for (int64_t dimension : getShape()) {
    if (dimension == VariableType::kDynamic) {
      printer << "?";
    } else {
      printer << dimension;
    }

    printer << "x";
  }

  printer << getElementType();

  if (isDiscrete()) {
    printer << ", discrete";
  } else if (isParameter()) {
    printer << ", parameter";
  } else if (isConstant()) {
    printer << ", constant";
  }

  if (isInput()) {
    printer << ", input";
  } else if (isOutput()) {
    printer << ", output";
  }

  printer << ">";
}

VariableType VariableType::get(llvm::ArrayRef<int64_t> shape,
                               mlir::Type elementType,
                               VariabilityProperty variabilityProperty,
                               IOProperty ioProperty,
                               mlir::Attribute memorySpace) {
  // Drop default memory space value and replace it with empty attribute.
  memorySpace = skipDefaultMemorySpace(memorySpace);

  return Base::get(elementType.getContext(), shape, elementType,
                   variabilityProperty, ioProperty, memorySpace);
}

VariableType VariableType::getChecked(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitErrorFn,
    llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
    VariabilityProperty variabilityProperty, IOProperty ioProperty,
    mlir::Attribute memorySpace) {
  // Drop default memory space value and replace it with empty attribute.
  memorySpace = skipDefaultMemorySpace(memorySpace);

  return Base::getChecked(emitErrorFn, elementType.getContext(), shape,
                          elementType, variabilityProperty, ioProperty,
                          memorySpace);
}

mlir::LogicalResult
VariableType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                     llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                     VariabilityProperty variabilityProperty,
                     IOProperty ioProperty, mlir::Attribute memorySpace) {
  if (!isValidElementType(elementType)) {
    return emitError() << "invalid variable element type";
  }

  // Negative sizes are not allowed.
  for (int64_t size : shape) {
    if (size < 0 && size != VariableType::kDynamic) {
      return emitError() << "invalid variable size";
    }
  }

  if (!isSupportedMemorySpace(memorySpace)) {
    return emitError() << "unsupported memory space Attribute";
  }

  return mlir::success();
}

bool VariableType::hasRank() const { return true; }

mlir::ShapedType
VariableType::cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                        mlir::Type elementType) const {
  VariableType::Builder builder(*shape, elementType);
  builder.setVariabilityProperty(getVariabilityProperty());
  builder.setVisibilityProperty(getVisibilityProperty());
  builder.setMemorySpace(getMemorySpace());
  return builder;
}

bool VariableType::isValidElementType(mlir::Type type) {
  return type.isIntOrIndexOrFloat() ||
         type.isa<BooleanType, IntegerType, RealType, RecordType>();
}

bool VariableType::isScalar() const { return getRank() == 0; }

bool VariableType::isDiscrete() const {
  return getVariabilityProperty() == VariabilityProperty::discrete;
}

bool VariableType::isParameter() const {
  return getVariabilityProperty() == VariabilityProperty::parameter;
}

bool VariableType::isConstant() const {
  return getVariabilityProperty() == VariabilityProperty::constant;
}

bool VariableType::isReadOnly() const { return isParameter() || isConstant(); }

bool VariableType::isInput() const {
  return getVisibilityProperty() == IOProperty::input;
}

bool VariableType::isOutput() const {
  return getVisibilityProperty() == IOProperty::output;
}

VariableType VariableType::wrap(mlir::Type type,
                                VariabilityProperty variabilityProperty,
                                IOProperty ioProperty) {
  if (auto arrayType = type.dyn_cast<ArrayType>()) {
    return VariableType::get(arrayType.getShape(), arrayType.getElementType(),
                             variabilityProperty, ioProperty,
                             arrayType.getMemorySpace());
  }

  return VariableType::get(std::nullopt, type, variabilityProperty, ioProperty);
}

ArrayType VariableType::toArrayType() const {
  return ArrayType::get(getShape(), getElementType(), getMemorySpace());
}

mlir::TensorType VariableType::toTensorType() const {
  return mlir::RankedTensorType::get(getShape(), getElementType());
}

mlir::Type VariableType::unwrap() const {
  if (!isScalar()) {
    return toTensorType();
  }

  return getElementType();
}

VariableType VariableType::withShape(llvm::ArrayRef<int64_t> shape) const {
  return VariableType::get(shape, getElementType(), getVariabilityProperty(),
                           getVisibilityProperty());
}

VariableType VariableType::withType(mlir::Type type) const {
  return VariableType::get(getShape(), type, getVariabilityProperty(),
                           getVisibilityProperty());
}

VariableType VariableType::withVariabilityProperty(
    VariabilityProperty variabilityProperty) const {
  return VariableType::get(getShape(), getElementType(), variabilityProperty,
                           getVisibilityProperty());
}

VariableType VariableType::withoutVariabilityProperty() const {
  return withVariabilityProperty(VariabilityProperty::none);
}

VariableType VariableType::asDiscrete() const {
  return withVariabilityProperty(VariabilityProperty::discrete);
}

VariableType VariableType::asParameter() const {
  return withVariabilityProperty(VariabilityProperty::parameter);
}

VariableType VariableType::asConstant() const {
  return withVariabilityProperty(VariabilityProperty::constant);
}

VariableType VariableType::withIOProperty(IOProperty ioProperty) const {
  return VariableType::get(getShape(), getElementType(),
                           getVariabilityProperty(), ioProperty);
}

VariableType VariableType::withoutIOProperty() const {
  return withIOProperty(IOProperty::none);
}

VariableType VariableType::asInput() const {
  return withIOProperty(IOProperty::input);
}

VariableType VariableType::asOutput() const {
  return withIOProperty(IOProperty::output);
}
} // namespace mlir::bmodelica

namespace mlir::bmodelica::detail {
bool isSupportedMemorySpace(mlir::Attribute memorySpace) {
  // Empty attribute is allowed as default memory space.
  if (!memorySpace) {
    return true;
  }

  // Supported built-in attributes.
  if (memorySpace
          .isa<mlir::IntegerAttr, mlir::StringAttr, mlir::DictionaryAttr>()) {
    return true;
  }

  // Allow custom dialect attributes.
  if (!isa<mlir::BuiltinDialect>(memorySpace.getDialect())) {
    return true;
  }

  return false;
}

mlir::Attribute skipDefaultMemorySpace(mlir::Attribute memorySpace) {
  mlir::IntegerAttr intMemorySpace =
      memorySpace.dyn_cast_or_null<mlir::IntegerAttr>();

  if (intMemorySpace && intMemorySpace.getValue() == 0) {
    return nullptr;
  }

  return memorySpace;
}
} // namespace mlir::bmodelica::detail
