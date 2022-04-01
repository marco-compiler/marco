#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Modelica/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace ::mlir::modelica;

namespace mlir::modelica
{
  //===----------------------------------------------------------------------===//
  // ArrayType
  //===----------------------------------------------------------------------===//

  mlir::Type ArrayType::parse(mlir::MLIRContext* context, mlir::DialectAsmParser& parser)
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
    return ArrayType::get(context, elementType, castedDims);
  }

  void ArrayType::print(DialectAsmPrinter& os) const
  {
    os << "array<";

    for (const auto& dimension : getShape()) {
      os << (dimension == -1 ? "?" : std::to_string(dimension)) << "x";
    }

    os << getElementType() << ">";
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
      if (size == -1) {
        return -1;
      }

      result *= size;
    }

    return result;
  }

  bool ArrayType::hasConstantShape() const
  {
    return llvm::all_of(getShape(), [](long size) {
      return size != -1;
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

  UnsizedArrayType ArrayType::toUnsized() const
  {
    return UnsizedArrayType::get(getContext(), getElementType());
  }

  bool ArrayType::canBeOnStack() const
  {
    return hasConstantShape();
  }

  //===----------------------------------------------------------------------===//
  // UnsizedArrayType
  //===----------------------------------------------------------------------===//

  mlir::Type UnsizedArrayType::parse(mlir::MLIRContext* context, mlir::DialectAsmParser& parser)
  {
    mlir::Type elementType;
    
    if (parser.parseLess() ||
        parser.parseKeyword("*") ||
        parser.parseKeyword("x") ||
        parser.parseType(elementType) ||
        parser.parseGreater()) {
      return mlir::Type();
    }
    
    return UnsizedArrayType::get(context, elementType);
  }

  void UnsizedArrayType::print(DialectAsmPrinter& os) const
  {
    os << "array<*x" << getElementType() << ">";
  }

  //===----------------------------------------------------------------------===//
  // MemberType
  //===----------------------------------------------------------------------===//

  mlir::Type MemberType::parse(mlir::MLIRContext* context, mlir::DialectAsmParser& parser)
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
    return MemberType::get(context, elementType, castedDimensions, isConstant, ioProperty);
  }

  void MemberType::print(DialectAsmPrinter& os) const
  {
    os << "member<";

    for (const auto& dimension : getShape()) {
      os << (dimension == -1 ? "?" : std::to_string(dimension)) << "x";
    }

    os << getElementType();

    if (isConstant()) {
      os << ", constant";
    }

    if (isInput()) {
      os << ", input";
    } else if (isOutput()) {
      os << ", output";
    }

    os << ">";
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

  MemberType MemberType::withIOProperty(IOProperty ioProperty) const
  {
    return MemberType::get(getContext(), getElementType(), getShape(), isConstant(), ioProperty);
  }
}
