#include "marco/Codegen/Utils.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

[[maybe_unused]] static bool checkArrayTypeCompatibility(ArrayType first, ArrayType second)
{
  if (first.getRank() != second.getRank()) {
    return false;
  }

  return llvm::all_of(llvm::zip(first.getShape(), second.getShape()), [](const auto& pair) {
    auto firstDimension = std::get<0>(pair);
    auto secondDimension = std::get<1>(pair);

    if (firstDimension == ArrayType::kDynamicSize || secondDimension == ArrayType::kDynamicSize) {
      return true;
    }

    return firstDimension == secondDimension;
  });
}

namespace marco::codegen
{
  mlir::Type getMostGenericType(mlir::Value x, mlir::Value y)
  {
    assert(x != nullptr && y != nullptr);
    return getMostGenericType(x.getType(), y.getType());
  }

  mlir::Type getMostGenericType(mlir::Type x, mlir::Type y)
  {
    assert((x.isa<BooleanType, IntegerType, RealType, mlir::IndexType>()));
    assert((y.isa<BooleanType, IntegerType, RealType, mlir::IndexType>()));

    if (x.isa<BooleanType>()) {
      return y;
    }

    if (y.isa<BooleanType>()) {
      return x;
    }

    if (x.isa<RealType>()) {
      return x;
    }

    if (y.isa<RealType>()) {
      return y;
    }

    if (x.isa<IntegerType>()) {
      return y;
    }

    return x;
  }
}
