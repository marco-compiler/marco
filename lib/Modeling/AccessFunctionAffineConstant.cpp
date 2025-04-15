#include "marco/Modeling/AccessFunctionAffineConstant.h"
#include "marco/Modeling/DimensionAccessConstant.h"
#include "marco/Modeling/DimensionAccessIndices.h"
#include "marco/Modeling/DimensionAccessRange.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling {
bool AccessFunctionAffineConstant::canBeBuilt(
    llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results) {
  if (results.empty()) {
    return true;
  }

  return llvm::all_of(results, [](const auto &result) {
    return result->template dyn_cast<DimensionAccessConstant>();
  });
}

bool AccessFunctionAffineConstant::canBeBuilt(mlir::AffineMap affineMap) {
  return llvm::all_of(affineMap.getResults(), [](mlir::AffineExpr expression) {
    return mlir::isa<mlir::AffineConstantExpr>(expression);
  });
}

AccessFunctionAffineConstant::AccessFunctionAffineConstant(
    mlir::AffineMap affineMap)
    : AccessFunctionAffineMap(Kind::Affine_Constant, affineMap) {
  assert(canBeBuilt(affineMap));
}

AccessFunctionAffineConstant::AccessFunctionAffineConstant(
    mlir::MLIRContext *context, uint64_t numOfDimensions,
    llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
    : AccessFunctionAffineConstant(
          buildAffineMap(context, numOfDimensions, results)) {}

AccessFunctionAffineConstant::~AccessFunctionAffineConstant() = default;

std::unique_ptr<AccessFunction> AccessFunctionAffineConstant::clone() const {
  return std::make_unique<AccessFunctionAffineConstant>(*this);
}

bool AccessFunctionAffineConstant::operator==(
    const AccessFunction &other) const {
  if (auto otherCasted = other.dyn_cast<AccessFunctionAffineConstant>()) {
    return *this == *otherCasted;
  }

  return false;
}

bool AccessFunctionAffineConstant::operator==(
    const AccessFunctionAffineConstant &other) const {
  return getAffineMap() == other.getAffineMap();
}

bool AccessFunctionAffineConstant::operator!=(
    const AccessFunction &other) const {
  return !(*this == other);
}

bool AccessFunctionAffineConstant::operator!=(
    const AccessFunctionAffineConstant &other) const {
  return !(*this == other);
}

bool AccessFunctionAffineConstant::isConstant() const { return true; }

IndexSet AccessFunctionAffineConstant::map(const IndexSet &indices) const {
  if (indices.empty() && getNumOfDims() != 0) {
    return {};
  }

  if (auto mappedPoint = getMappedPoint()) {
    return {*mappedPoint};
  }

  return {};
}

IndexSet
AccessFunctionAffineConstant::inverseMap(const IndexSet &accessedIndices,
                                         const IndexSet &parentIndices) const {
  auto expectedAccessedIndices = map(parentIndices);

  if (accessedIndices.empty() && !expectedAccessedIndices.empty()) {
    return {};
  }

  if (expectedAccessedIndices.contains(accessedIndices)) {
    return parentIndices;
  }

  return {};
}

std::optional<Point> AccessFunctionAffineConstant::getMappedPoint() const {
  mlir::AffineMap affineMap = getAffineMap();

  if (affineMap.getNumResults() == 0) {
    return std::nullopt;
  }

  llvm::SmallVector<Point::data_type> coordinates(affineMap.getNumResults(), 0);

  for (size_t i = 0, e = affineMap.getNumResults(); i < e; ++i) {
    auto constantExpr =
        mlir::cast<mlir::AffineConstantExpr>(affineMap.getResult(i));

    coordinates[i] = constantExpr.getValue();
  }

  return Point(coordinates);
}
} // namespace marco::modeling
