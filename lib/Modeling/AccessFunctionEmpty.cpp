#include "marco/Modeling/AccessFunctionEmpty.h"

using namespace ::marco::modeling;

namespace marco::modeling {
bool AccessFunctionEmpty::canBeBuilt(
    uint64_t numOfDimensions,
    llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results) {
  return numOfDimensions == 0 && results.empty();
}

bool AccessFunctionEmpty::canBeBuilt(mlir::AffineMap affineMap) {
  return affineMap.getNumDims() == 0 && affineMap.getNumResults() == 0;
}

AccessFunctionEmpty::AccessFunctionEmpty(mlir::AffineMap affineMap)
    : AccessFunctionAffineMap(Kind::Empty, affineMap) {
  assert(canBeBuilt(affineMap));
}

AccessFunctionEmpty::AccessFunctionEmpty(
    mlir::MLIRContext *context, uint64_t numOfDimensions,
    llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
    : AccessFunctionEmpty(buildAffineMap(context, numOfDimensions, results)) {}

AccessFunctionEmpty::~AccessFunctionEmpty() = default;

std::unique_ptr<AccessFunction> AccessFunctionEmpty::clone() const {
  return std::make_unique<AccessFunctionEmpty>(*this);
}

bool AccessFunctionEmpty::operator==(const AccessFunction &other) const {
  if (auto otherCasted = other.dyn_cast<AccessFunctionEmpty>()) {
    return *this == *otherCasted;
  }

  return false;
}

bool AccessFunctionEmpty::operator==(const AccessFunctionEmpty &other) const {
  return true;
}

bool AccessFunctionEmpty::operator!=(const AccessFunction &other) const {
  return !(*this == other);
}

bool AccessFunctionEmpty::operator!=(const AccessFunctionEmpty &other) const {
  return false;
}

bool AccessFunctionEmpty::isInvertible() const { return true; }

std::unique_ptr<AccessFunction> AccessFunctionEmpty::inverse() const {
  return clone();
}

IndexSet AccessFunctionEmpty::map(const Point &point) const { return {}; }

IndexSet AccessFunctionEmpty::map(const IndexSet &indices) const { return {}; }

IndexSet AccessFunctionEmpty::inverseMap(const IndexSet &accessedIndices,
                                         const IndexSet &parentIndices) const {
  assert(accessedIndices.empty());
  assert(parentIndices.empty());
  return {};
}
} // namespace marco::modeling
