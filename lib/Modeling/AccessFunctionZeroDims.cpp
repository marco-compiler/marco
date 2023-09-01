#include "marco/Modeling/AccessFunctionZeroDims.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  AccessFunctionZeroDims::AccessFunctionZeroDims(
      mlir::AffineMap affineMap)
      : AccessFunction(
          AccessFunction::Kind::ZeroDims,
          affineMap)
  {
    assert(canBeBuilt(affineMap));
  }

  AccessFunctionZeroDims::~AccessFunctionZeroDims() = default;

  bool AccessFunctionZeroDims::canBeBuilt(mlir::AffineMap affineMap)
  {
    return affineMap.getNumDims() == 0;
  }

  std::unique_ptr<AccessFunction> AccessFunctionZeroDims::clone() const
  {
    return std::make_unique<AccessFunctionZeroDims>(*this);
  }

  IndexSet AccessFunctionZeroDims::map(const IndexSet& indices) const
  {
    auto extendedAccessFunction = getWithAtLeastNDimensions(1);

    llvm::SmallVector<Range, 3> dummyRanges(
        extendedAccessFunction->getNumOfDims(), Range(0, 1));

    return extendedAccessFunction->map(
        IndexSet(MultidimensionalRange(dummyRanges)));
  }
}
