#include "marco/Modeling/AccessFunctionEmpty.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  AccessFunctionEmpty::AccessFunctionEmpty(
      mlir::AffineMap affineMap)
      : AccessFunction(
          AccessFunction::Kind::Empty,
          affineMap)
  {
    assert(canBeBuilt(affineMap));
  }

  AccessFunctionEmpty::~AccessFunctionEmpty() = default;

  bool AccessFunctionEmpty::canBeBuilt(mlir::AffineMap affineMap)
  {
    return affineMap.isEmpty();
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
