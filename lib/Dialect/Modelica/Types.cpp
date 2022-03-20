#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Modelica/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace ::mlir::modelica;

static ArrayAllocationScope memberToArrayAllocationScope(MemberAllocationScope scope)
{
  switch (scope) {
    case MemberAllocationScope::stack:
      return ArrayAllocationScope::stack;

    case MemberAllocationScope::heap:
      return ArrayAllocationScope::heap;
  }

  llvm_unreachable("Unknown member allocation scope");
  return ArrayAllocationScope::heap;
}

static MemberAllocationScope arrayToMemberAllocationScope(ArrayAllocationScope scope)
{
  switch (scope) {
    case ArrayAllocationScope::stack:
      return MemberAllocationScope::stack;

    case ArrayAllocationScope::heap:
    case ArrayAllocationScope::unknown:
      return MemberAllocationScope::heap;
  }

  llvm_unreachable("Unknown array allocation scope");
  return MemberAllocationScope::heap;
}

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

    auto allocationScope = ArrayAllocationScope::unknown;

    if (mlir::succeeded(parser.parseKeyword("heap"))) {
      allocationScope = ArrayAllocationScope::heap;

      if (parser.parseComma()) {
        return mlir::Type();
      }

    } else if (mlir::succeeded(parser.parseKeyword("stack"))) {
      allocationScope = ArrayAllocationScope::stack;

      if (parser.parseComma()) {
        return mlir::Type();
      }
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
    return ArrayType::get(context, allocationScope, elementType, castedDims);
  }

  void ArrayType::print(DialectAsmPrinter& os) const
  {
    os << "array<";
    auto allocationScope = getAllocationScope();

    if (allocationScope == ArrayAllocationScope::heap) {
      os << "heap, ";
    } else if (allocationScope == ArrayAllocationScope::stack) {
      os << "stack, ";
    }

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

    return ArrayType::get(getContext(), getAllocationScope(), getElementType(), resultShape);
  }

  ArrayType ArrayType::toAllocationScope(ArrayAllocationScope scope) const
  {
    return ArrayType::get(getContext(), scope, getElementType(), getShape());
  }

  ArrayType ArrayType::toUnknownAllocationScope() const
  {
    return toAllocationScope(ArrayAllocationScope::unknown);
  }

  ArrayType ArrayType::toMinAllowedAllocationScope() const
  {
    if (getAllocationScope() == ArrayAllocationScope::heap) {
      return *this;
    }

    if (canBeOnStack()) {
      return toAllocationScope(ArrayAllocationScope::stack);
    }

    return toAllocationScope(ArrayAllocationScope::heap);
  }

  ArrayType ArrayType::toElementType(mlir::Type elementType) const
  {
    return ArrayType::get(getContext(), getAllocationScope(), elementType, getShape());
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

    auto allocationScope = MemberAllocationScope::heap;

    if (mlir::succeeded(parser.parseKeyword("heap"))) {
      allocationScope = MemberAllocationScope::heap;

      if (parser.parseComma()) {
        return mlir::Type();
      }
    } else {
      allocationScope = MemberAllocationScope::stack;

      if (parser.parseKeyword("stack") ||
          parser.parseComma()) {
        return mlir::Type();
      }
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
    return MemberType::get(context, allocationScope, elementType, castedDims);
  }

  void MemberType::print(DialectAsmPrinter& os) const
  {
    os << "member<";
    auto allocationScope = getAllocationScope();

    if (allocationScope == MemberAllocationScope::stack) {
      os << "stack, ";
    } else if (allocationScope == MemberAllocationScope::heap) {
      os << "heap, ";
    }

    for (const auto& dimension : getShape()) {
      os << (dimension == -1 ? "?" : std::to_string(dimension)) << "x";
    }

    os << getElementType() << ">";
  }

  unsigned int MemberType::getRank() const
  {
    return getShape().size();
  }

  MemberType MemberType::wrap(mlir::Type type, MemberAllocationScope allocationScope)
  {
    if (auto arrayType = type.dyn_cast<ArrayType>()) {
      return MemberType::get(type.getContext(), allocationScope, arrayType.getElementType(), arrayType.getShape());
    }

    return MemberType::get(type.getContext(), allocationScope, type, llvm::None);
  }

  ArrayType MemberType::toArrayType() const
  {
    return ArrayType::get(
        getContext(),
        memberToArrayAllocationScope(getAllocationScope()),
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
}
