#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/AccessFunctionConstant.h"
#include "marco/Modeling/AccessFunctionEmpty.h"
#include "marco/Modeling/AccessFunctionGeneric.h"
#include "marco/Modeling/AccessFunctionRotoTranslation.h"
#include "marco/Modeling/DimensionAccessConstant.h"
#include "marco/Modeling/DimensionAccessDimension.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

static std::unique_ptr<AccessFunction> build(
    mlir::MLIRContext* context,
    unsigned int numOfDimensions,
    llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results,
    DimensionAccess::FakeDimensionsMap fakeDimensionsMap)
{
  if (fakeDimensionsMap.empty()) {
    if (AccessFunctionEmpty::canBeBuilt(numOfDimensions, results)) {
      return std::make_unique<AccessFunctionEmpty>(
          context, numOfDimensions, results);
    }

    if (AccessFunctionConstant::canBeBuilt(results)) {
      return std::make_unique<AccessFunctionConstant>(
          context, numOfDimensions, results);
    }

    if (AccessFunctionRotoTranslation::canBeBuilt(numOfDimensions, results)) {
      return std::make_unique<AccessFunctionRotoTranslation>(
          context, numOfDimensions, results);
    }
  }

  // Fallback implementation.
  return std::make_unique<AccessFunctionGeneric>(
      context, numOfDimensions, results, std::move(fakeDimensionsMap));
}

namespace marco::modeling
{
  std::unique_ptr<AccessFunction>
  AccessFunction::build(
      mlir::MLIRContext* context,
      unsigned int numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
  {
    llvm::SmallVector<mlir::AffineExpr> affineExpressions;
    DimensionAccess::FakeDimensionsMap fakeDimensionsMap;

    for (const auto& result : results) {
      affineExpressions.push_back(
          result->getAffineExpr(numOfDimensions, fakeDimensionsMap));
    }

    for (auto& expression : affineExpressions) {
      expression = mlir::simplifyAffineExpr(
          expression, numOfDimensions + fakeDimensionsMap.size(), 0);
    }

    auto affineMap = mlir::AffineMap::get(
        numOfDimensions + fakeDimensionsMap.size(), 0,
        affineExpressions, context);

    return AccessFunction::fromExtendedMap(affineMap, fakeDimensionsMap);
  }

  std::unique_ptr<AccessFunction>
  AccessFunction::build(mlir::AffineMap affineMap)
  {
    assert(affineMap.getNumSymbols() == 0);
    mlir::AffineMap simplifiedAffineMap = mlir::simplifyAffineMap(affineMap);
    DimensionAccess::FakeDimensionsMap fakeDimensionsMap;

    return ::build(
        simplifiedAffineMap.getContext(), simplifiedAffineMap.getNumDims(),
        convertAffineExpressions(simplifiedAffineMap.getResults()),
        std::move(fakeDimensionsMap));
  }

  std::unique_ptr<AccessFunction> AccessFunction::fromExtendedMap(
      mlir::AffineMap affineMap,
      const DimensionAccess::FakeDimensionsMap& fakeDimensionsMap)
  {
    mlir::AffineMap simplifiedAffineMap = mlir::simplifyAffineMap(affineMap);
    llvm::SmallVector<std::unique_ptr<DimensionAccess>> results;

    for (mlir::AffineExpr result : simplifiedAffineMap.getResults()) {
      results.push_back(
          DimensionAccess::getDimensionAccessFromExtendedMap(
              result, fakeDimensionsMap));
    }

    size_t numOfFakeDimensions = fakeDimensionsMap.size();

    return ::build(
        simplifiedAffineMap.getContext(),
        simplifiedAffineMap.getNumDims() - numOfFakeDimensions,
        results, fakeDimensionsMap);
  }

  llvm::SmallVector<std::unique_ptr<DimensionAccess>>
  AccessFunction::convertAffineExpressions(
      llvm::ArrayRef<mlir::AffineExpr> expressions)
  {
    llvm::SmallVector<std::unique_ptr<DimensionAccess>> results;

    for (mlir::AffineExpr expression : expressions) {
      results.push_back(DimensionAccess::build(expression));
    }

    return results;
  }

  AccessFunction::AccessFunction(
      AccessFunction::Kind kind, mlir::MLIRContext* context)
      : kind(kind),
        context(context)
  {
  }

  AccessFunction::AccessFunction(const AccessFunction& other) = default;

  AccessFunction::~AccessFunction() = default;

  void swap(AccessFunction& first, AccessFunction& second)
  {
    using std::swap;
    swap(first.kind, second.kind);
    swap(first.context, second.context);
  }

  mlir::MLIRContext* AccessFunction::getContext() const
  {
    return context;
  }

  bool AccessFunction::isAffine() const
  {
    return false;
  }

  mlir::AffineMap AccessFunction::getAffineMap() const
  {
    llvm_unreachable("Not implemented");
    return {};
  }

  bool AccessFunction::isIdentity() const
  {
    return false;
  }

  std::unique_ptr<AccessFunction>
  AccessFunction::combine(const AccessFunction& other) const
  {
    auto otherAFWithGivenDims =
        other.getWithGivenDimensions(getNumOfResults());

    if (isAffine() && otherAFWithGivenDims->isAffine()) {
      return AccessFunction::build(
          otherAFWithGivenDims->getAffineMap().compose(getAffineMap()));
    }

    // The inputs count of the right-hand side must be equal to the results of
    // the left-hand side.
    auto rhsWithExtraDimensions =
        other.getWithGivenDimensions(getNumOfResults());

    size_t rhsExtraDimensions =
        rhsWithExtraDimensions->getNumOfDims() - other.getNumOfDims();

    // Get the affine map of the right-hand side and keep track of the
    // additional dimensions added in the process.
    DimensionAccess::FakeDimensionsMap rhsFakeDimensionsMap;

    mlir::AffineMap rhsAffineMap =
        rhsWithExtraDimensions->getExtendedAffineMap(rhsFakeDimensionsMap);

    size_t rhsFakeDimensionsCount = rhsFakeDimensionsMap.size();

    // The fake dimensions of the right-hand side must be added to the
    // left-hand side and forwarded as results.
    auto lhsWithExtraDimensions = this->getWithGivenDimensions(
        getNumOfDims() + rhsFakeDimensionsCount);

    llvm::SmallVector<std::unique_ptr<DimensionAccess>> lhsExtendedResults;

    for (const auto& result : getGeneralizedAccesses()) {
      lhsExtendedResults.push_back(result->clone());
    }

    for (size_t i = 0; i < rhsFakeDimensionsCount; ++i) {
      lhsExtendedResults.push_back(
          std::make_unique<DimensionAccessDimension>(
              getContext(), getNumOfDims() + i));
    }

    auto lhsWithExtraResults = AccessFunction::build(
        getContext(), lhsWithExtraDimensions->getNumOfDims(),
        lhsExtendedResults);

    // Get the left-hand side affine map.
    DimensionAccess::FakeDimensionsMap lhsFakeDimensionsMap;

    auto lhsAffineMap =
        lhsWithExtraResults->getExtendedAffineMap(lhsFakeDimensionsMap);

    // Compose the accesses.
    mlir::AffineMap combinedAffineMap = rhsAffineMap.compose(lhsAffineMap);

    // Determine the fake dimensions of the result.
    DimensionAccess::FakeDimensionsMap resultFakeDimensionsMap;

    for (size_t i = 0; i < rhsFakeDimensionsCount; ++i) {
      size_t rhsDimension = other.getNumOfDims() + rhsExtraDimensions + i;
      size_t lhsDimension = getNumOfDims() + i;

      resultFakeDimensionsMap[lhsDimension] =
          rhsFakeDimensionsMap[rhsDimension];
    }

    // Remove the additional dimensions.
    return AccessFunction::fromExtendedMap(
        combinedAffineMap, resultFakeDimensionsMap);
  }

  bool AccessFunction::isInvertible() const
  {
    return false;
  }

  std::unique_ptr<AccessFunction> AccessFunction::inverse() const
  {
    return nullptr;
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

  std::unique_ptr<AccessFunction> AccessFunction::getWithGivenDimensions(
      uint64_t requestedDims) const
  {
    return AccessFunction::build(
        getContext(), std::max(requestedDims, getNumOfDims()),
        getGeneralizedAccesses());
  }

  llvm::raw_ostream& operator<<(
      llvm::raw_ostream& os, const AccessFunction& obj)
  {
    return obj.dump(os);
  }
}
