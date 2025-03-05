#include "marco/Dialect/BaseModelica/IR/Attributes.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::bmodelica;

namespace mlir {
template <>
struct FieldParser<mlir::bmodelica::EquationPath> {
  static FailureOr<mlir::bmodelica::EquationPath>
  parse(mlir::AsmParser &parser) {
    EquationPath::EquationSide side = EquationPath::LEFT;
    llvm::SmallVector<uint64_t> path;

    if (parser.parseLSquare()) {
      return mlir::failure();
    }

    if (mlir::succeeded(parser.parseOptionalKeyword("R"))) {
      side = EquationPath::RIGHT;
    } else {
      if (parser.parseKeyword("L")) {
        return mlir::failure();
      }
    }

    while (mlir::succeeded(parser.parseOptionalComma())) {
      int64_t index;

      if (parser.parseInteger(index)) {
        return mlir::failure();
      }

      path.push_back(index);
    }

    return EquationPath(side, path);
  }
};

mlir::AsmPrinter &operator<<(mlir::AsmPrinter &printer,
                             const mlir::bmodelica::EquationPath &path) {
  printer.getStream() << path;
  return printer;
}

template <>
struct FieldParser<mlir::bmodelica::EquationScheduleDirection> {
  static FailureOr<mlir::bmodelica::EquationScheduleDirection>
  parse(mlir::AsmParser &parser) {
    EquationScheduleDirection direction = EquationScheduleDirection::Unknown;

    if (mlir::succeeded(parser.parseOptionalKeyword("any"))) {
      direction = EquationScheduleDirection::Any;
    } else if (mlir::succeeded(parser.parseOptionalKeyword("forward"))) {
      direction = EquationScheduleDirection::Forward;
    } else if (mlir::succeeded(parser.parseOptionalKeyword("backward"))) {
      direction = EquationScheduleDirection::Backward;
    } else if (parser.parseKeyword("unknown")) {
      return mlir::failure();
    }

    return direction;
  }
};

mlir::AsmPrinter &
operator<<(mlir::AsmPrinter &printer,
           const mlir::bmodelica::EquationScheduleDirection &direction) {
  switch (direction) {
  case EquationScheduleDirection::Any:
    printer << "any";
    break;

  case EquationScheduleDirection::Forward:
    printer << "forward";
    break;

  case EquationScheduleDirection::Backward:
    printer << "backward";
    break;

  case EquationScheduleDirection::Unknown:
    printer << "unknown";
    break;
  }

  return printer;
}
} // namespace mlir

//===----------------------------------------------------------------------===//
// Tablegen attribute definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/BaseModelica/IR/BaseModelicaAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// BaseModelicaDialect
//===----------------------------------------------------------------------===//

namespace mlir::bmodelica {
void BaseModelicaDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "marco/Dialect/BaseModelica/IR/BaseModelicaAttributes.cpp.inc"
      >();
}
} // namespace mlir::bmodelica

namespace mlir::bmodelica {
mlir::Attribute getAttr(mlir::Type type, int64_t value) {
  if (type.isa<BooleanType>()) {
    return BooleanAttr::get(type.getContext(), value != 0);
  }

  if (type.isa<IntegerType>()) {
    return IntegerAttr::get(type.getContext(), value);
  }

  if (type.isa<RealType>()) {
    return RealAttr::get(type.getContext(), static_cast<double>(value));
  }

  if (type.isa<mlir::IndexType>()) {
    return mlir::IntegerAttr::get(type, value);
  }

  if (type.isa<mlir::IntegerType>()) {
    return mlir::IntegerAttr::get(type, value);
  }

  if (type.isa<mlir::FloatType>()) {
    return mlir::FloatAttr::get(type, static_cast<double>(value));
  }

  llvm_unreachable("Unknown Modelica type");
  return {};
}

mlir::Attribute getAttr(mlir::Type type, double value) {
  if (type.isa<BooleanType>()) {
    return BooleanAttr::get(type.getContext(), value != 0);
  }

  if (type.isa<IntegerType>()) {
    return IntegerAttr::get(type.getContext(), static_cast<int64_t>(value));
  }

  if (type.isa<RealType>()) {
    return RealAttr::get(type.getContext(), value);
  }

  if (type.isa<mlir::IndexType>()) {
    return mlir::IntegerAttr::get(type, static_cast<int64_t>(value));
  }

  if (type.isa<mlir::IntegerType>()) {
    return mlir::IntegerAttr::get(type, static_cast<int64_t>(value));
  }

  if (type.isa<mlir::FloatType>()) {
    return mlir::FloatAttr::get(type, value);
  }

  llvm_unreachable("Unknown Modelica type");
  return {};
}

mlir::Attribute getAttr(ArrayType arrayType, llvm::ArrayRef<int64_t> values) {
  mlir::Type elementType = arrayType.getElementType();

  if (elementType.isa<BooleanType>()) {
    llvm::SmallVector<bool> casted;

    for (int64_t value : values) {
      casted.push_back(value != 0);
    }

    return DenseBooleanElementsAttr::get(arrayType, casted);
  }

  if (elementType.isa<IntegerType>()) {
    return DenseIntegerElementsAttr::get(arrayType, values);
  }

  if (elementType.isa<RealType>()) {
    llvm::SmallVector<double> casted;

    for (int64_t value : values) {
      casted.push_back(static_cast<double>(value));
    }

    return DenseRealElementsAttr::get(arrayType, casted);
  }

  if (elementType.isa<mlir::IndexType>()) {
    return DenseIntegerElementsAttr::get(arrayType, values);
  }

  if (elementType.isa<mlir::IntegerType>()) {
    return DenseIntegerElementsAttr::get(arrayType, values);
  }

  if (elementType.isa<mlir::FloatType>()) {
    llvm::SmallVector<double> casted;

    for (int64_t value : values) {
      casted.push_back(static_cast<double>(value));
    }

    return DenseRealElementsAttr::get(arrayType, casted);
  }

  llvm_unreachable("Unknown Modelica array type");
  return {};
}

mlir::Attribute getAttr(ArrayType arrayType, llvm::ArrayRef<double> values) {
  mlir::Type elementType = arrayType.getElementType();

  if (elementType.isa<BooleanType>()) {
    llvm::SmallVector<bool> casted;

    for (double value : values) {
      casted.push_back(value != 0);
    }

    return DenseBooleanElementsAttr::get(arrayType, casted);
  }

  if (elementType.isa<IntegerType>()) {
    llvm::SmallVector<int64_t> casted;

    for (double value : values) {
      casted.push_back(static_cast<int64_t>(value));
    }

    return DenseIntegerElementsAttr::get(arrayType, casted);
  }

  if (elementType.isa<RealType>()) {
    return DenseRealElementsAttr::get(arrayType, values);
  }

  if (elementType.isa<mlir::IndexType>()) {
    llvm::SmallVector<int64_t> casted;

    for (double value : values) {
      casted.push_back(static_cast<int64_t>(value));
    }

    return DenseIntegerElementsAttr::get(arrayType, casted);
  }

  if (elementType.isa<mlir::IntegerType>()) {
    llvm::SmallVector<int64_t> casted;

    for (double value : values) {
      casted.push_back(static_cast<int64_t>(value));
    }

    return DenseIntegerElementsAttr::get(arrayType, casted);
  }

  if (elementType.isa<mlir::FloatType>()) {
    return DenseRealElementsAttr::get(arrayType, values);
  }

  llvm_unreachable("Unknown Modelica array type");
  return {};
}

//===-------------------------------------------------------------------===//
// BooleanAttr
//===-------------------------------------------------------------------===//

mlir::Attribute BooleanAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  bool value;

  if (mlir::succeeded(parser.parseOptionalKeyword("true"))) {
    value = true;
  } else {
    if (parser.parseKeyword("false")) {
      return {};
    }

    value = false;
  }

  if (!type) {
    type = BooleanType::get(parser.getContext());
  }

  return BooleanAttr::get(parser.getContext(), type, value);
}

void BooleanAttr::print(mlir::AsmPrinter &printer) const {
  printer << " " << (getValue() == 0 ? "false" : "true");
}

//===-------------------------------------------------------------------===//
// IntegerAttr
//===-------------------------------------------------------------------===//

mlir::Attribute IntegerAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  int64_t value;

  if (parser.parseInteger(value)) {
    return {};
  }

  if (!type) {
    type = IntegerType::get(parser.getContext());
  }

  return IntegerAttr::get(parser.getContext(), type,
                          llvm::APInt(64, value, true));
}

void IntegerAttr::print(mlir::AsmPrinter &printer) const {
  printer << " " << getValue();
}

//===-------------------------------------------------------------------===//
// RealAttr
//===-------------------------------------------------------------------===//

mlir::Attribute RealAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  double value;

  if (parser.parseFloat(value)) {
    return {};
  }

  if (!type) {
    type = RealType::get(parser.getContext());
  }

  return RealAttr::get(parser.getContext(), type, llvm::APFloat(value));
}

void RealAttr::print(mlir::AsmPrinter &printer) const {
  printer << " " << getValue();
}

//===----------------------------------------------------------------------===//
// DenseBooleanElementsAttr
//===----------------------------------------------------------------------===//

mlir::Attribute DenseBooleanElementsAttr::parse(mlir::AsmParser &parser,
                                                mlir::Type type) {
  llvm::SmallVector<bool> values;

  if (parser.parseLess() || parser.parseLSquare()) {
    return {};
  }

  bool value = false;

  if (mlir::succeeded(parser.parseOptionalKeyword("true"))) {
    value = true;
  } else if (parser.parseKeyword("false")) {
    return {};
  }

  values.push_back(value);

  while (mlir::succeeded(parser.parseOptionalComma())) {
    if (mlir::succeeded(parser.parseOptionalKeyword("true"))) {
      value = true;
    } else if (parser.parseKeyword("false")) {
      return {};
    }

    values.push_back(value);
  }

  if (parser.parseRSquare() || parser.parseGreater()) {
    return {};
  }

  if (!type) {
    llvm::SmallVector<int64_t, 1> shape;
    shape.push_back(static_cast<int64_t>(values.size()));

    type = mlir::RankedTensorType::get(shape,
                                       BooleanType::get(parser.getContext()));
  }

  return DenseBooleanElementsAttr::get(parser.getContext(), type, values);
}

void DenseBooleanElementsAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<[";

  for (const auto &value : llvm::enumerate(getValues())) {
    if (value.index() != 0) {
      printer << ", ";
    }

    printer << (value.value() ? "true" : "false");
  }

  printer << "]> : " << getType();
}

//===----------------------------------------------------------------------===//
// DenseIntegerElementsAttr
//===----------------------------------------------------------------------===//

mlir::Attribute DenseIntegerElementsAttr::parse(mlir::AsmParser &parser,
                                                mlir::Type type) {
  llvm::SmallVector<int64_t> values;

  if (parser.parseLess() || parser.parseLSquare()) {
    return {};
  }

  int64_t value;

  if (parser.parseInteger(value)) {
    return {};
  }

  values.push_back(value);

  while (mlir::succeeded(parser.parseOptionalComma())) {
    if (parser.parseInteger(value)) {
      return {};
    }

    values.push_back(value);
  }

  if (parser.parseRSquare() || parser.parseGreater()) {
    return {};
  }

  if (!type) {
    llvm::SmallVector<int64_t, 1> shape;
    shape.push_back(static_cast<int64_t>(values.size()));

    type = mlir::RankedTensorType::get(shape,
                                       IntegerType::get(parser.getContext()));
  }

  return DenseIntegerElementsAttr::get(type, values);
}

void DenseIntegerElementsAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<[";

  for (const auto &value : llvm::enumerate(getValues())) {
    if (value.index() != 0) {
      printer << ", ";
    }

    printer << value.value();
  }

  printer << "]> : " << getType();
}

//===----------------------------------------------------------------------===//
// DenseRealElementsAttr
//===----------------------------------------------------------------------===//

mlir::Attribute DenseRealElementsAttr::parse(mlir::AsmParser &parser,
                                             mlir::Type type) {
  llvm::SmallVector<llvm::APFloat> values;

  if (parser.parseLess() || parser.parseLSquare()) {
    return {};
  }

  double value;

  if (parser.parseFloat(value)) {
    return {};
  }

  values.emplace_back(value);

  while (mlir::succeeded(parser.parseOptionalComma())) {
    if (parser.parseFloat(value)) {
      return {};
    }

    values.emplace_back(value);
  }

  if (parser.parseRSquare() || parser.parseGreater()) {
    return {};
  }

  if (!type) {
    llvm::SmallVector<int64_t, 1> shape;
    shape.push_back(static_cast<int64_t>(values.size()));

    type =
        mlir::RankedTensorType::get(shape, RealType::get(parser.getContext()));
  }

  return DenseRealElementsAttr::get(parser.getContext(), type, values);
}

void DenseRealElementsAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<[";

  for (const auto &value : llvm::enumerate(getValues())) {
    if (value.index() != 0) {
      printer << ", ";
    }

    printer << value.value();
  }

  printer << "]> : " << getType();
}

//===----------------------------------------------------------------------===//
// InverseFunctionsAttr
//===----------------------------------------------------------------------===//

bool InverseFunctionsMap::operator==(const InverseFunctionsMap &other) const {
  return map == other.map;
}

InverseFunctionsMap::InverseFunction &
InverseFunctionsMap::operator[](unsigned int arg) {
  return map[arg];
}

bool InverseFunctionsMap::empty() const { return map.empty(); }

InverseFunctionsMap::iterator InverseFunctionsMap::begin() {
  return map.begin();
}

InverseFunctionsMap::const_iterator InverseFunctionsMap::begin() const {
  return map.begin();
}

InverseFunctionsMap::iterator InverseFunctionsMap::end() { return map.end(); }

InverseFunctionsMap::const_iterator InverseFunctionsMap::end() const {
  return map.end();
}

InverseFunctionsMap InverseFunctionsMap::allocateInto(
    mlir::StorageUniquer::StorageAllocator &allocator) {
  InverseFunctionsMap result;

  for (const auto &entry : map) {
    result.map[entry.first] =
        std::make_pair(allocator.copyInto(entry.second.first),
                       allocator.copyInto(entry.second.second));
  }

  return result;
}

bool InverseFunctionsMap::isInvertible(unsigned int argumentIndex) const {
  return map.find(argumentIndex) != map.end();
}

llvm::StringRef
InverseFunctionsMap::getFunction(unsigned int argumentIndex) const {
  return map.find(argumentIndex)->second.first;
}

llvm::ArrayRef<unsigned int>
InverseFunctionsMap::getArgumentsIndexes(unsigned int argumentIndex) const {
  return map.find(argumentIndex)->second.second;
}

llvm::hash_code hash_value(const InverseFunctionsMap &map) {
  return llvm::hash_combine_range(map.begin(), map.end());
}

mlir::Attribute InverseFunctionsAttr::parse(mlir::AsmParser &parser,
                                            mlir::Type) {
  // TODO parse InverseFunctionsAttr
  llvm_unreachable("InverseFunctionsAttr parsing is not implemented");
  return mlir::Attribute();
}

void InverseFunctionsAttr::print(mlir::AsmPrinter &printer) const {
  printer << "inverse<";
  auto inverseFunctionsMap = getInverseFunctionsMap();
  bool separator = false;

  for (auto invertibleArg : inverseFunctionsMap) {
    if (separator) {
      printer << ", ";
    }

    printer << invertibleArg << ": ";
    printer << inverseFunctionsMap.getFunction(invertibleArg) << "(";
    bool innerSeparator = false;

    for (const auto &arg :
         inverseFunctionsMap.getArgumentsIndexes(invertibleArg)) {
      if (innerSeparator) {
        printer << ", ";
      }

      printer << arg;
      innerSeparator = true;
    }

    printer << ")";
    separator = true;
  }

  printer << ">";
}

bool InverseFunctionsAttr::isInvertible(unsigned int argumentIndex) const {
  return getInverseFunctionsMap().isInvertible(argumentIndex);
}

llvm::StringRef
InverseFunctionsAttr::getFunction(unsigned int argumentIndex) const {
  return getInverseFunctionsMap().getFunction(argumentIndex);
}

llvm::ArrayRef<unsigned int>
InverseFunctionsAttr::getArgumentsIndexes(unsigned int argumentIndex) const {
  return getInverseFunctionsMap().getArgumentsIndexes(argumentIndex);
}

//===----------------------------------------------------------------------===//
// FunctionDerivativeAttr
//===----------------------------------------------------------------------===//

//===-------------------------------------------------------------------===//
// IntegerRangeAttr
//===-------------------------------------------------------------------===//

IntegerRangeAttr IntegerRangeAttr::get(mlir::MLIRContext *context,
                                       int64_t lowerBound, int64_t upperBound,
                                       int64_t step) {
  auto type = RangeType::get(context, IntegerType::get(context));
  return get(context, type, lowerBound, upperBound, step);
}

mlir::Attribute IntegerRangeAttr::parse(mlir::AsmParser &parser,
                                        mlir::Type type) {
  int64_t lowerBound;
  int64_t upperBound;
  int64_t step;

  if (parser.parseLess() || parser.parseInteger(lowerBound) ||
      parser.parseComma() || parser.parseInteger(upperBound) ||
      parser.parseComma() || parser.parseInteger(step) ||
      parser.parseGreater()) {
    return {};
  }

  if (!type) {
    type = RangeType::get(parser.getContext(),
                          IntegerType::get(parser.getContext()));
  }

  return IntegerRangeAttr::get(parser.getContext(), type, lowerBound,
                               upperBound, step);
}

void IntegerRangeAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getLowerBound() << ", " << getUpperBound() << ", "
          << getStep() << ">";
}

int64_t IntegerRangeAttr::getNumOfElements() const {
  return 1 + (getUpperBound() - getLowerBound()) / getStep();
}

//===-------------------------------------------------------------------===//
// RealRangeAttr
//===-------------------------------------------------------------------===//

RealRangeAttr RealRangeAttr::get(mlir::Type type, double lowerBound,
                                 double upperBound, double step) {
  return get(type.getContext(), type, llvm::APFloat(lowerBound),
             llvm::APFloat(upperBound), llvm::APFloat(step));
}

RealRangeAttr RealRangeAttr::get(mlir::MLIRContext *context, double lowerBound,
                                 double upperBound, double step) {
  auto type = RangeType::get(context, RealType::get(context));

  return get(context, type, llvm::APFloat(lowerBound),
             llvm::APFloat(upperBound), llvm::APFloat(step));
}

mlir::Attribute RealRangeAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  double lowerBound;
  double upperBound;
  double step;

  if (parser.parseLess() || parser.parseFloat(lowerBound) ||
      parser.parseComma() || parser.parseFloat(upperBound) ||
      parser.parseComma() || parser.parseFloat(step) || parser.parseGreater()) {
    return {};
  }

  if (!type) {
    type =
        RangeType::get(parser.getContext(), RealType::get(parser.getContext()));
  }

  return RealRangeAttr::get(parser.getContext(), type,
                            llvm::APFloat(lowerBound),
                            llvm::APFloat(upperBound), llvm::APFloat(step));
}

void RealRangeAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getLowerBound() << ", " << getUpperBound() << ", "
          << getStep() << ">";
}

int64_t RealRangeAttr::getNumOfElements() const {
  llvm::APFloat result = ((getUpperBound() - getLowerBound()) / getStep());
  return 1 + static_cast<int64_t>(result.convertToDouble());
}
} // namespace mlir::bmodelica
