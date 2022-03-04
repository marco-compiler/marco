#include "marco/Dialect/Modelica/ModelicaBuilder.h"

namespace mlir::modelica
{
  ModelicaBuilder::ModelicaBuilder(mlir::MLIRContext* context)
      : mlir::OpBuilder(context)
  {
  }

  BooleanType ModelicaBuilder::getBooleanType()
  {
    return BooleanType::get(getContext());
  }

  IntegerType ModelicaBuilder::getIntegerType()
  {
    return IntegerType::get(getContext());
  }

  RealType ModelicaBuilder::getRealType()
  {
    return RealType::get(getContext());
  }

  ArrayType ModelicaBuilder::getArrayType(
      ArrayAllocationScope allocationScope, mlir::Type elementType, llvm::ArrayRef<long> shape)
  {
    return ArrayType::get(getContext(), allocationScope, elementType, shape);
  }

  BooleanAttr ModelicaBuilder::getBooleanAttribute(bool value)
  {
    return BooleanAttr::get(getBooleanType(), value);
  }

  IntegerAttr ModelicaBuilder::getIntegerAttribute(long value)
  {
    return IntegerAttr::get(getIntegerType(), value);
  }

  RealAttr ModelicaBuilder::getRealAttribute(double value)
  {
    return RealAttr::get(getRealType(), value);
  }

  mlir::Attribute ModelicaBuilder::getZeroAttribute(mlir::Type type)
  {
    if (type.isa<BooleanType>()) {
      return getBooleanAttribute(false);
    }

    if (type.isa<IntegerType>()) {
      return getIntegerAttribute(0);
    }

    if (type.isa<RealType>()) {
      return getRealAttribute(0);
    }

    return getZeroAttr(type);
  }
}
