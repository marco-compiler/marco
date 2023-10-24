#include "marco/Modeling/AccessFunctionZeroResults.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  AccessFunctionZeroResults::AccessFunctionZeroResults(
      mlir::MLIRContext* context,
      unsigned int numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
      : AccessFunction(
          AccessFunction::Kind::ZeroResults,
          context, numOfDimensions, results)
  {
    assert(canBeBuilt(numOfDimensions, results));
  }

  AccessFunctionZeroResults::AccessFunctionZeroResults(
      mlir::AffineMap affineMap)
      : AccessFunction(affineMap.getContext(),
                       affineMap.getNumDims(),
                       convertAffineExpressions(affineMap.getResults()))
  {
  }

  AccessFunctionZeroResults::~AccessFunctionZeroResults() = default;

  bool AccessFunctionZeroResults::canBeBuilt(
      unsigned int numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
  {
    return results.empty();
  }

  bool AccessFunctionZeroResults::canBeBuilt(mlir::AffineMap affineMap)
  {
    llvm::SmallVector<std::unique_ptr<DimensionAccess>> results;
    AccessFunction::convertAffineExpressions(affineMap.getResults());
    return AccessFunctionZeroResults::canBeBuilt(affineMap.getNumDims(), results);
  }

  std::unique_ptr<AccessFunction> AccessFunctionZeroResults::clone() const
  {
    return std::make_unique<AccessFunctionZeroResults>(*this);
  }

  IndexSet AccessFunctionZeroResults::map(const IndexSet& indices) const
  {
    return {};
  }

  IndexSet AccessFunctionZeroResults::inverseMap(
      const IndexSet& accessedIndices,
      const IndexSet& parentIndices) const
  {
    return parentIndices;
  }
}
