#include "marco/Modeling/AccessFunctionEmpty.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  AccessFunctionEmpty::AccessFunctionEmpty(
      mlir::MLIRContext* context,
      unsigned int numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
      : AccessFunction(
          AccessFunction::Kind::Empty,
          context, numOfDimensions, results)
  {
    assert(canBeBuilt(numOfDimensions, results));
  }

  AccessFunctionEmpty::AccessFunctionEmpty(mlir::AffineMap affineMap)
      : AccessFunctionEmpty(
          affineMap.getContext(),
          affineMap.getNumDims(),
          convertAffineExpressions(affineMap.getResults()))
  {
  }

  AccessFunctionEmpty::~AccessFunctionEmpty() = default;

  bool AccessFunctionEmpty::canBeBuilt(
      unsigned int numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
  {
    return numOfDimensions == 0 && results.empty();
  }

  bool AccessFunctionEmpty::canBeBuilt(mlir::AffineMap affineMap)
  {
    return AccessFunctionEmpty::canBeBuilt(
        affineMap.getNumDims(),
        AccessFunction::convertAffineExpressions(affineMap.getResults()));
  }

  std::unique_ptr<AccessFunction> AccessFunctionEmpty::clone() const
  {
    return std::make_unique<AccessFunctionEmpty>(*this);
  }

  bool AccessFunctionEmpty::isInvertible() const
  {
    return true;
  }

  std::unique_ptr<AccessFunction> AccessFunctionEmpty::inverse() const
  {
    return clone();
  }

  IndexSet AccessFunctionEmpty::map(const IndexSet& indices) const
  {
    return {};
  }
}
