#include "marco/Modeling/AccessFunctionAffineMap.h"

using namespace ::marco::modeling;

namespace marco::modeling {
mlir::AffineMap AccessFunctionAffineMap::buildAffineMap(
    mlir::MLIRContext *context, uint64_t numOfDimensions,
    llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results) {
  llvm::SmallVector<mlir::AffineExpr, 6> expressions;

  for (const auto &result : results) {
    expressions.push_back(result->getAffineExpr());
  }

  return mlir::AffineMap::get(numOfDimensions, 0, expressions, context);
}

AccessFunctionAffineMap::AccessFunctionAffineMap(Kind kind,
                                                 mlir::AffineMap affineMap)
    : AccessFunction(kind, affineMap.getContext()), affineMap(affineMap) {}

llvm::raw_ostream &AccessFunctionAffineMap::dump(llvm::raw_ostream &os) const {
  return os << affineMap;
}

uint64_t AccessFunctionAffineMap::getNumOfDims() const {
  return affineMap.getNumDims();
}

uint64_t AccessFunctionAffineMap::getNumOfResults() const {
  return affineMap.getNumResults();
}

bool AccessFunctionAffineMap::isConstant() const {
  return affineMap.isConstant();
}

bool AccessFunctionAffineMap::isAffine() const { return true; }

mlir::AffineMap AccessFunctionAffineMap::getAffineMap() const {
  return affineMap;
}

bool AccessFunctionAffineMap::isIdentity() const {
  return affineMap.isIdentity();
}

IndexSet AccessFunctionAffineMap::map(const Point &point) const {
  mlir::AffineMap map = getAffineMapWithGivenDimensions(point.rank());
  auto resultCoordinates = map.compose(point);

  llvm::SmallVector<Point::data_type, 6> result(resultCoordinates.begin(),
                                                resultCoordinates.end());

  return {Point(result)};
}

IndexSet AccessFunctionAffineMap::map(const IndexSet &indices) const {
  IndexSet mappedIndices;

  for (Point point : indices) {
    mappedIndices += map(point);
  }

  return mappedIndices;
}

std::unique_ptr<AccessFunction>
AccessFunctionAffineMap::getWithGivenDimensions(uint64_t requestedDims) const {
  return AccessFunction::build(getAffineMapWithGivenDimensions(requestedDims));
}

llvm::SmallVector<std::unique_ptr<DimensionAccess>, 6>
AccessFunctionAffineMap::getGeneralizedAccesses() const {
  return convertAffineExpressions(affineMap.getResults());
}

mlir::AffineMap AccessFunctionAffineMap::getExtendedAffineMap(
    DimensionAccess::FakeDimensionsMap &fakeDimensionsMap) const {
  return getAffineMap();
}

mlir::AffineMap AccessFunctionAffineMap::getAffineMapWithGivenDimensions(
    uint64_t requestedDims) const {
  uint64_t resultDimensions =
      std::max(static_cast<uint64_t>(affineMap.getNumDims()), requestedDims);

  return mlir::AffineMap::get(resultDimensions, 0, affineMap.getResults(),
                              affineMap.getContext());
}
} // namespace marco::modeling
