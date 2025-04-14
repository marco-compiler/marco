#include "marco/Modeling/AccessFunctionConstant.h"
#include "marco/Modeling/DimensionAccessConstant.h"
#include "marco/Modeling/DimensionAccessIndices.h"
#include "marco/Modeling/DimensionAccessRange.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling {
bool AccessFunctionConstant::canBeBuilt(
    llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results) {
  if (results.empty()) {
    return true;
  }

  return llvm::all_of(results, [](const auto &result) {
    return result->template dyn_cast<DimensionAccessConstant>() ||
           result->template dyn_cast<DimensionAccessRange>() ||
           result->template dyn_cast<DimensionAccessIndices>();
  });
}

bool AccessFunctionConstant::canBeBuilt(mlir::AffineMap affineMap) {
  return llvm::all_of(affineMap.getResults(), [](mlir::AffineExpr expression) {
    return mlir::isa<mlir::AffineConstantExpr>(expression);
  });
}

AccessFunctionConstant::AccessFunctionConstant(
    mlir::MLIRContext *context, uint64_t numOfDimensions,
    llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
    : AccessFunctionGeneric(Kind::Constant, context, numOfDimensions, results) {
  assert(canBeBuilt(results));
}

AccessFunctionConstant::AccessFunctionConstant(mlir::AffineMap affineMap)
    : AccessFunctionConstant(affineMap.getContext(), affineMap.getNumDims(),
                             convertAffineExpressions(affineMap.getResults())) {
}

AccessFunctionConstant::~AccessFunctionConstant() = default;

std::unique_ptr<AccessFunction> AccessFunctionConstant::clone() const {
  return std::make_unique<AccessFunctionConstant>(*this);
}

IndexSet AccessFunctionConstant::map(const IndexSet &indices) const {
  if (indices.empty() && getNumOfDims() != 0) {
    return {};
  }

  llvm::SmallVector<Point::data_type> dummyCoordinates(getNumOfDims(), 0);
  Point dummyPoint(dummyCoordinates);
  return map(dummyPoint);
}

IndexSet
AccessFunctionConstant::inverseMap(const IndexSet &accessedIndices,
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
} // namespace marco::modeling
