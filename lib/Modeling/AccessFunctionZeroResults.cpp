#include "marco/Modeling/AccessFunctionZeroResults.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  AccessFunctionZeroResults::AccessFunctionZeroResults(
      mlir::AffineMap affineMap)
      : AccessFunction(
          AccessFunction::Kind::ZeroResults,
          affineMap)
  {
    assert(canBeBuilt(affineMap));
  }

  AccessFunctionZeroResults::~AccessFunctionZeroResults() = default;

  bool AccessFunctionZeroResults::canBeBuilt(mlir::AffineMap affineMap)
  {
    return affineMap.getNumResults() == 0;
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
