#include "marco/Modeling/AccessFunctionConstant.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  AccessFunctionConstant::AccessFunctionConstant(
      mlir::AffineMap affineMap)
      : AccessFunction(
          AccessFunction::Kind::Constant,
          affineMap)
  {
    assert(canBeBuilt(affineMap));
  }

  AccessFunctionConstant::~AccessFunctionConstant() = default;

  bool AccessFunctionConstant::canBeBuilt(mlir::AffineMap affineMap)
  {
    if (affineMap.getNumResults() == 0) {
      return false;
    }

    return llvm::all_of(
        affineMap.getResults(),
        [](mlir::AffineExpr expression) {
          return expression.isa<mlir::AffineConstantExpr>();
        });
  }

  std::unique_ptr<AccessFunction> AccessFunctionConstant::clone() const
  {
    return std::make_unique<AccessFunctionConstant>(*this);
  }

  IndexSet AccessFunctionConstant::map(const IndexSet& indices) const
  {
    auto extendedAccessFunction = getWithAtLeastNDimensions(1);

    llvm::SmallVector<Point::data_type, 3> dummyCoordinates(
        extendedAccessFunction->getNumOfDims(), 0);

    return IndexSet(extendedAccessFunction->map(Point(dummyCoordinates)));
  }

  IndexSet AccessFunctionConstant::inverseMap(
      const IndexSet& accessedIndices,
      const IndexSet& parentIndices) const
  {
    return parentIndices;
  }
}
