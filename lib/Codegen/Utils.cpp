#include "marco/Codegen/Utils.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"

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

  std::string getUniqueSymbolName(mlir::ModuleOp module, std::function<std::string(void)> tryFn)
  {
    std::string name = tryFn();

    while (module.lookupSymbol(name) != nullptr) {
      name = tryFn();
    }

    return name;
  }

  void copyArray(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value source, mlir::Value destination)
  {
    assert(source.getType().isa<ArrayType>());
    assert(destination.getType().isa<ArrayType>());

    auto sourceArrayType = source.getType().cast<ArrayType>();
    [[maybe_unused]] auto destinationArrayType = destination.getType().cast<ArrayType>();

    assert(checkArrayTypeCompatibility(sourceArrayType, destinationArrayType));

    auto rank = sourceArrayType.getRank();

    mlir::Value zero = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(0));
    mlir::Value one = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(1));

    llvm::SmallVector<mlir::Value, 3> lowerBounds(rank, zero);
    llvm::SmallVector<mlir::Value, 3> upperBounds;
    llvm::SmallVector<mlir::Value, 3> steps(rank, one);

    for (unsigned int i = 0, e = sourceArrayType.getRank(); i < e; ++i) {
      mlir::Value dim = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(i));
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

  modeling::IndexSet getIndexSet(mlir::modelica::IterationSpace iterationSpace)
  {
    IndexSet current;

    if (auto iterationRange = iterationSpace.getIterationRange(); iterationRange.step() == 1) {
      current += MultidimensionalRange(Range(iterationRange.from(), iterationRange.to() + 1));
    } else {
      for (auto index = iterationRange.from(); index < iterationRange.to(); index += iterationRange.step()) {
        current += Point(index);
      }
    }

    if (!iterationSpace.hasSubDimensions()) {
      return current;
    }

    modeling::IndexSet result;

    for (const auto& subDimension : iterationSpace) {
      auto extension = current.intersect(MultidimensionalRange(Range(
          subDimension.first.from(), subDimension.first.to() + 1)));

      for (const auto& childRange : getIndexSet(*subDimension.second)) {
        for (const auto& parentRange : extension) {
          std::vector<Range> extended;

          for (size_t i = 0; i < parentRange.rank(); ++i) {
            extended.push_back(parentRange[i]);
          }

          for (size_t i = 0; i < childRange.rank(); ++i) {
            extended.push_back(childRange[i]);
          }

          result += MultidimensionalRange(extended);
        }
      }
    }

    return result;
  }
}
