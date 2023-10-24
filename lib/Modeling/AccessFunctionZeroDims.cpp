#include "marco/Modeling/AccessFunctionZeroDims.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  AccessFunctionZeroDims::AccessFunctionZeroDims(
      mlir::MLIRContext* context,
      unsigned int numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
      : AccessFunction(
          AccessFunction::Kind::ZeroDims,
          context, numOfDimensions, results)
  {
    assert(canBeBuilt(numOfDimensions, results));
  }

  AccessFunctionZeroDims::AccessFunctionZeroDims(mlir::AffineMap affineMap)
      : AccessFunction(affineMap.getContext(),
                       affineMap.getNumDims(),
                       convertAffineExpressions(affineMap.getResults()))
  {
  }

  AccessFunctionZeroDims::~AccessFunctionZeroDims() = default;

  bool AccessFunctionZeroDims::canBeBuilt(
      unsigned int numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
  {
    return numOfDimensions == 0;
  }

  bool AccessFunctionZeroDims::canBeBuilt(mlir::AffineMap affineMap)
  {
    llvm::SmallVector<std::unique_ptr<DimensionAccess>> results;
    AccessFunction::convertAffineExpressions(affineMap.getResults());
    return AccessFunctionZeroDims::canBeBuilt(affineMap.getNumDims(), results);
  }

  std::unique_ptr<AccessFunction> AccessFunctionZeroDims::clone() const
  {
    return std::make_unique<AccessFunctionZeroDims>(*this);
  }

  IndexSet AccessFunctionZeroDims::map(const IndexSet& indices) const
  {
    Point dummyPoint(0);
    IndexSet mappedIndices;

    for (const auto& result : getResults()) {
      mappedIndices = mappedIndices.append(result->map(dummyPoint));
    }

    return mappedIndices;
  }
}
