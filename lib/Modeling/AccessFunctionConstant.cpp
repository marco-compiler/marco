#include "marco/Modeling/AccessFunctionConstant.h"
#include "marco/Modeling/DimensionAccessConstant.h"
#include "marco/Modeling/DimensionAccessIndices.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  AccessFunctionConstant::AccessFunctionConstant(
      mlir::MLIRContext* context,
      unsigned int numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
      : AccessFunction(
          AccessFunction::Kind::Constant,
          context, numOfDimensions, results)
  {
    assert(canBeBuilt(numOfDimensions, results));
  }

  AccessFunctionConstant::AccessFunctionConstant(mlir::AffineMap affineMap)
      : AccessFunctionConstant(
          affineMap.getContext(),
          affineMap.getNumDims(),
          convertAffineExpressions(affineMap.getResults()))
  {
  }

  AccessFunctionConstant::~AccessFunctionConstant() = default;

  bool AccessFunctionConstant::canBeBuilt(
      unsigned int numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
  {
    if (results.empty()) {
      return false;
    }

    return llvm::all_of(results, [](const auto& result) {
      return result->template dyn_cast<DimensionAccessConstant>() ||
          result->template dyn_cast<DimensionAccessIndices>();
    });
  }

  bool AccessFunctionConstant::canBeBuilt(mlir::AffineMap affineMap)
  {
    return AccessFunctionConstant::canBeBuilt(
        affineMap.getNumDims(),
        AccessFunction::convertAffineExpressions(affineMap.getResults()));
  }

  std::unique_ptr<AccessFunction> AccessFunctionConstant::clone() const
  {
    return std::make_unique<AccessFunctionConstant>(*this);
  }

  IndexSet AccessFunctionConstant::map(const IndexSet& indices) const
  {
    llvm::SmallVector<Point::data_type, 6> dummyCoordinates(getNumOfDims(), 0);
    Point dummyPoint(dummyCoordinates);

    IndexSet mappedIndices;

    for (const auto& result : getResults()) {
      mappedIndices = mappedIndices.append(
          result->map(dummyPoint, getFakeDimensionsMap()));
    }

    return mappedIndices;
  }

  IndexSet AccessFunctionConstant::inverseMap(
      const IndexSet& accessedIndices,
      const IndexSet& parentIndices) const
  {
    return parentIndices;
  }
}
