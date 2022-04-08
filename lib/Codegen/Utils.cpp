#include "marco/Codegen/Utils.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"

using namespace ::marco;
using namespace ::marco::codegen;
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
  void copyArray(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value source, mlir::Value destination)
  {
    assert(source.getType().isa<ArrayType>());
    assert(destination.getType().isa<ArrayType>());

    auto sourceArrayType = source.getType().cast<ArrayType>();
    auto destinationArrayType = destination.getType().cast<ArrayType>();

    assert(checkArrayTypeCompatibility(sourceArrayType, destinationArrayType));

    auto rank = sourceArrayType.getRank();

    mlir::Value zero = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(0));
    mlir::Value one = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(1));

    llvm::SmallVector<mlir::Value, 3> lowerBounds(rank, zero);
    llvm::SmallVector<mlir::Value, 3> upperBounds;
    llvm::SmallVector<mlir::Value, 3> steps(rank, one);

    for (unsigned int i = 0, e = sourceArrayType.getRank(); i < e; ++i) {
      mlir::Value dim = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(i));
      upperBounds.push_back(builder.create<DimOp>(loc, source, dim));
    }

    // Create nested loops in order to iterate on each dimension of the array
    mlir::scf::buildLoopNest(
        builder, loc, lowerBounds, upperBounds, steps,
        [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange indices) {
          mlir::Value value = nestedBuilder.create<LoadOp>(loc, source, indices);
          nestedBuilder.create<StoreOp>(loc, value, destination, indices);
        });
  }
}
