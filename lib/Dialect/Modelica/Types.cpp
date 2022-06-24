#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Modelica/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::modelica;

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

//===----------------------------------------------------------------------===//
// Modelica Types
//===----------------------------------------------------------------------===//

namespace mlir::modelica
{
  /*
  bool ModelicaType::classof(mlir::Type type)
  {
    return llvm::isa<ModelicaDialect>(type.getDialect());
  }
   */
}

namespace mlir::modelica
{
  //===----------------------------------------------------------------------===//
  // ArrayType
  //===----------------------------------------------------------------------===//

  mlir::Type ArrayType::parse(mlir::AsmParser& parser)
  {
    if (parser.parseLess()) {
      return mlir::Type();
    }

    llvm::SmallVector<int64_t, 3> dimensions;

    if (parser.parseDimensionList(dimensions)) {
      return mlir::Type();
    }

    Type elementType;

    if (parser.parseType(elementType) ||
        parser.parseGreater()) {
      return mlir::Type();
    }

    llvm::SmallVector<long, 3> castedDims(dimensions.begin(), dimensions.end());
    return ArrayType::get(parser.getContext(), elementType, castedDims);
  }

  void ArrayType::print(mlir::AsmPrinter& printer) const
  {
    printer << "<";

    for (const auto& dimension : getShape()) {
      printer << (dimension == ArrayType::kDynamicSize ? "?" : std::to_string(dimension)) << "x";
    }

    printer << getElementType() << ">";
  }

  unsigned int ArrayType::getRank() const
  {
    return getShape().size();
  }

  unsigned int ArrayType::getConstantDimensionsCount() const
  {
    return llvm::count_if(getShape(), [](const auto& dimension) {
      return dimension > 0;
    });
  }

  unsigned int ArrayType::getDynamicDimensionsCount() const
  {
    return llvm::count_if(getShape(), [](const auto& dimension) {
      return dimension == -1;
    });
  }

  long ArrayType::getFlatSize() const
  {
    long result = 1;

    for (long size : getShape()) {
      if (size == ArrayType::kDynamicSize) {
        return ArrayType::kDynamicSize;
      }

      result *= size;
    }

    return result;
  }

  bool ArrayType::hasConstantShape() const
  {
    return llvm::all_of(getShape(), [](long size) {
      return size != ArrayType::kDynamicSize;
    });
  }

  bool ArrayType::isScalar() const
  {
    return getRank() == 0;
  }

  ArrayType ArrayType::slice(unsigned int subscriptsAmount) const
  {
    auto shape = getShape();
    assert(subscriptsAmount <= shape.size() && "Too many subscriptions");
    llvm::SmallVector<long, 3> resultShape;

    for (size_t i = subscriptsAmount, e = shape.size(); i < e; ++i) {
      resultShape.push_back(shape[i]);
    }

    return ArrayType::get(getContext(), getElementType(), resultShape);
  }

  ArrayType ArrayType::toElementType(mlir::Type elementType) const
  {
    return ArrayType::get(getContext(), elementType, getShape());
  }

  bool ArrayType::canBeOnStack() const
  {
    return hasConstantShape();
  }

  //===----------------------------------------------------------------------===//
  // MemberType
  //===----------------------------------------------------------------------===//

  mlir::Type MemberType::parse(mlir::AsmParser& parser)
  {
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

    bool isConstant = false;
    IOProperty ioProperty = IOProperty::none;

    while (mlir::succeeded(parser.parseOptionalComma())) {
      if (mlir::succeeded(parser.parseOptionalKeyword("constant"))) {
        isConstant = true;
      } else if (mlir::succeeded(parser.parseOptionalKeyword("input"))) {
        ioProperty = IOProperty::input;
      } else if (mlir::succeeded(parser.parseOptionalKeyword("output"))) {
        ioProperty = IOProperty::output;
      }
    }

    if (parser.parseGreater()) {
      return mlir::Type();
    }

    llvm::SmallVector<long, 3> castedDimensions(dimensions.begin(), dimensions.end());
    return MemberType::get(parser.getContext(), elementType, castedDimensions, isConstant, ioProperty);
  }

  void MemberType::print(mlir::AsmPrinter& printer) const
  {
    printer << "<";

    for (const auto& dimension : getShape()) {
      printer << (dimension == -1 ? "?" : std::to_string(dimension)) << "x";
    }

    printer << getElementType();

    if (isConstant()) {
      printer << ", constant";
    }

    if (isInput()) {
      printer << ", input";
    } else if (isOutput()) {
      printer << ", output";
    }

    printer << ">";
  }

  unsigned int MemberType::getRank() const
  {
    return getShape().size();
  }

  MemberType MemberType::wrap(mlir::Type type, bool isConstant, IOProperty ioProperty)
  {
    if (auto arrayType = type.dyn_cast<ArrayType>()) {
      return MemberType::get(type.getContext(), arrayType.getElementType(), arrayType.getShape(), isConstant, ioProperty);
    }

    return MemberType::get(type.getContext(), type, llvm::None, isConstant, ioProperty);
  }

  ArrayType MemberType::toArrayType() const
  {
    return ArrayType::get(
        getContext(),
        getElementType(),
        getShape());
  }

  mlir::Type MemberType::unwrap() const
  {
    if (getRank() == 0) {
      return getElementType();
    }

    return toArrayType();
  }

  MemberType MemberType::withShape(llvm::ArrayRef<long> shape) const
  {
    return MemberType::get(getContext(), getElementType(), shape, isConstant(), getVisibilityProperty());
  }

  MemberType MemberType::withType(mlir::Type type) const
  {
    return MemberType::wrap(type, isConstant(), getVisibilityProperty());
  }

  MemberType MemberType::withIOProperty(IOProperty ioProperty) const
  {
    return MemberType::get(getContext(), getElementType(), getShape(), isConstant(), ioProperty);
  }
}
