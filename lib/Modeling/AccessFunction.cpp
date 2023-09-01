#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/AccessFunctionConstant.h"
#include "marco/Modeling/AccessFunctionEmpty.h"
#include "marco/Modeling/AccessFunctionRotoTranslation.h"
#include "marco/Modeling/AccessFunctionZeroDims.h"
#include "marco/Modeling/AccessFunctionZeroResults.h"

namespace marco::modeling
{
  std::unique_ptr<AccessFunction>
  AccessFunction::build(mlir::AffineMap affineMap)
  {
    affineMap = mlir::simplifyAffineMap(affineMap);

    if (AccessFunctionEmpty::canBeBuilt(affineMap)) {
      return std::make_unique<AccessFunctionEmpty>(affineMap);
    }

    if (AccessFunctionZeroDims::canBeBuilt(affineMap)) {
      return std::make_unique<AccessFunctionZeroDims>(affineMap);
    }

    if (AccessFunctionZeroResults::canBeBuilt(affineMap)) {
      return std::make_unique<AccessFunctionZeroResults>(affineMap);
    }

    if (AccessFunctionConstant::canBeBuilt(affineMap)) {
      return std::make_unique<AccessFunctionConstant>(affineMap);
    }

    if (AccessFunctionRotoTranslation::canBeBuilt(affineMap)) {
      return std::make_unique<AccessFunctionRotoTranslation>(affineMap);
    }

    // Fallback implementation.
    return std::make_unique<AccessFunction>(affineMap);
  }

  AccessFunction::AccessFunction(mlir::AffineMap affineMap)
      : AccessFunction(Kind::Generic, affineMap)
  {
  }

  AccessFunction::AccessFunction(
      AccessFunction::Kind kind, mlir::AffineMap affineMap)
      : kind(kind),
        affineMap(affineMap)
  {
  }

  AccessFunction::~AccessFunction() = default;

  std::unique_ptr<AccessFunction> AccessFunction::clone() const
  {
    return std::make_unique<AccessFunction>(*this);
  }

  bool AccessFunction::operator==(const AccessFunction& other) const
  {
    return affineMap == other.affineMap;
  }

  bool AccessFunction::operator!=(const AccessFunction& other) const
  {
    return affineMap != other.affineMap;
  }

  mlir::AffineMap AccessFunction::getAffineMap() const
  {
    return affineMap;
  }

  size_t AccessFunction::getNumOfDims() const
  {
    return affineMap.getNumDims();
  }

  size_t AccessFunction::getNumOfResults() const
  {
    return affineMap.getNumResults();
  }

  bool AccessFunction::isIdentity() const
  {
    return affineMap.isIdentity();
  }

  std::unique_ptr<AccessFunction>
  AccessFunction::combine(const AccessFunction& other) const
  {
    mlir::AffineMap otherMap =
        other.getAffineMapWithAtLeastNDimensions(affineMap.getNumResults());

    return AccessFunction::build(otherMap.compose(affineMap));
  }

  bool AccessFunction::isInvertible() const
  {
    return false;
  }

  std::unique_ptr<AccessFunction> AccessFunction::inverse() const
  {
    return nullptr;
  }

  Point AccessFunction::map(const Point& indices) const
  {
    mlir::AffineMap map = getAffineMapWithAtLeastNDimensions(indices.rank());
    return {map.compose(indices)};
  }

  IndexSet AccessFunction::map(const IndexSet& indices) const
  {
    IndexSet result;

    for (Point point : indices) {
      result += map(point);
    }

    return result;
  }

  IndexSet AccessFunction::inverseMap(
      const IndexSet& accessedIndices,
      const IndexSet& parentIndices) const
  {
    IndexSet result;

    for (const Point& point : parentIndices) {
      if (accessedIndices.contains(map(point))) {
        result += point;
      }
    }

    return result;
  }

  std::unique_ptr<AccessFunction>
  AccessFunction::getWithAtLeastNDimensions(unsigned int dimensions) const
  {
    return AccessFunction::build(
        getAffineMapWithAtLeastNDimensions(dimensions));
  }

  mlir::AffineMap
  AccessFunction::getAffineMapWithAtLeastNDimensions(unsigned int dimensions) const
  {
    mlir::AffineMap map = getAffineMap();

    if (map.getNumDims() >= dimensions) {
      return map;
    }

    return mlir::AffineMap::get(
        std::max(map.getNumDims(), dimensions),
        map.getNumSymbols(),
        map.getResults(),
        map.getContext());
  }
}
