#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/AccessFunctionConstant.h"
#include "marco/Modeling/AccessFunctionEmpty.h"
#include "marco/Modeling/AccessFunctionRotoTranslation.h"
#include "marco/Modeling/AccessFunctionZeroDims.h"
#include "marco/Modeling/AccessFunctionZeroResults.h"
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

    if (AccessFunctionZeroDims::canBeBuilt(numOfDimensions, results)) {
      return std::make_unique<AccessFunctionZeroDims>(
          context, numOfDimensions, results);
    }

    if (AccessFunctionZeroResults::canBeBuilt(numOfDimensions, results)) {
      return std::make_unique<AccessFunctionZeroResults>(
          context, numOfDimensions, results);
    }

    if (AccessFunctionConstant::canBeBuilt(numOfDimensions, results)) {
      return std::make_unique<AccessFunctionConstant>(
          context, numOfDimensions, results);
    }

    if (AccessFunctionRotoTranslation::canBeBuilt(numOfDimensions, results)) {
      return std::make_unique<AccessFunctionRotoTranslation>(
          context, numOfDimensions, results);
    }
  }

  // Fallback implementation.
  return std::make_unique<AccessFunction>(
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
      mlir::MLIRContext* context,
      unsigned int numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results,
      DimensionAccess::FakeDimensionsMap fakeDimensionsMap)
      : AccessFunction(Kind::Generic, context, numOfDimensions, results,
                       std::move(fakeDimensionsMap))
  {
  }

  AccessFunction::AccessFunction(mlir::AffineMap affineMap)
      : AccessFunction(affineMap.getContext(),
                       affineMap.getNumDims(),
                       convertAffineExpressions(affineMap.getResults()),
                       DimensionAccess::FakeDimensionsMap())
  {
  }

  AccessFunction::AccessFunction(
      AccessFunction::Kind kind,
      mlir::MLIRContext* context,
      unsigned int numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results,
      DimensionAccess::FakeDimensionsMap fakeDimensionsMap)
      : kind(kind),
        context(context),
        numOfDimensions(numOfDimensions),
        fakeDimensionsMap(std::move(fakeDimensionsMap))
  {
    for (const auto& result : results) {
      this->results.push_back(result->clone());
    }
  }

  AccessFunction::AccessFunction(const AccessFunction& other)
      : kind(other.kind),
        context(other.context),
        numOfDimensions(other.numOfDimensions),
        fakeDimensionsMap(other.fakeDimensionsMap)
  {
    for (const auto& result : other.results) {
      results.push_back(result->clone());
    }
  }

  AccessFunction::~AccessFunction() = default;

  AccessFunction& AccessFunction::operator=(const AccessFunction& other)
  {
    AccessFunction result(other);
    swap(*this, result);
    return *this;
  }

  AccessFunction& AccessFunction::operator=(AccessFunction&& other) = default;

  void swap(AccessFunction& first, AccessFunction& second)
  {
    using std::swap;
    swap(first.kind, second.kind);
    swap(first.context, second.context);
    swap(first.numOfDimensions, second.numOfDimensions);

    llvm::SmallVector<std::unique_ptr<DimensionAccess>> firstTmp =
        std::move(first.results);

    first.results = std::move(second.results);
    second.results = std::move(firstTmp);

    swap(first.fakeDimensionsMap, second.fakeDimensionsMap);
  }

  std::unique_ptr<AccessFunction> AccessFunction::clone() const
  {
    return std::make_unique<AccessFunction>(*this);
  }

  bool AccessFunction::operator==(const AccessFunction& other) const
  {
    if (numOfDimensions != other.numOfDimensions) {
      return false;
    }

    for (const auto& [lhs, rhs] : llvm::zip(results, other.results)) {
      if (*lhs != *rhs) {
        return false;
      }
    }

    return true;
  }

  bool AccessFunction::operator!=(const AccessFunction& other) const
  {
    if (numOfDimensions != other.numOfDimensions) {
      return true;
    }

    for (const auto& [lhs, rhs] : llvm::zip(results, other.results)) {
      if (*lhs != *rhs) {
        return true;
      }
    }

    return false;
  }

  llvm::raw_ostream& AccessFunction::dump(llvm::raw_ostream& os) const
  {
    os << "(";

    for (size_t i = 0, e = getNumOfDims(); i < e; ++i) {
      if (i != 0) {
        os << ", ";
      }

      os << "d" << i;
    }

    os << ") -> (";

    for (size_t i = 0, e = getNumOfResults(); i < e; ++i) {
      if (i != 0) {
        os << ", ";
      }

      os << *getResults()[i];
    }

    os << ")";
    return os;
  }

  mlir::MLIRContext* AccessFunction::getContext() const
  {
    return context;
  }

  size_t AccessFunction::getNumOfDims() const
  {
    return numOfDimensions;
  }

  size_t AccessFunction::getNumOfResults() const
  {
    return results.size();
  }

  llvm::ArrayRef<std::unique_ptr<DimensionAccess>>
  AccessFunction::getResults() const
  {
    return results;
  }

  bool AccessFunction::isAffine() const
  {
    return llvm::all_of(
        getResults(),
        [](const std::unique_ptr<DimensionAccess>& result) {
          return result->isAffine();
        });
  }

  mlir::AffineMap AccessFunction::getAffineMap() const
  {
    assert(isAffine());
    llvm::SmallVector<mlir::AffineExpr> expressions;

    for (const auto& result : getResults()) {
      expressions.push_back(result->getAffineExpr());
    }

    return mlir::AffineMap::get(getNumOfDims(), 0, expressions, getContext());
  }

  bool AccessFunction::isIdentity() const
  {
    unsigned int dimension = 0;

    for (const auto& result : getResults()) {
      if (!result->isa<DimensionAccessDimension>()) {
        return false;
      }

      auto dimAccess = result->cast<DimensionAccessDimension>();

      if (dimAccess->getDimension() != dimension++) {
        return false;
      }
    }

    return dimension == getNumOfDims();
  }

  std::unique_ptr<AccessFunction>
  AccessFunction::combine(const AccessFunction& other) const
  {
    // The inputs count of the right-hand side must be equal to the results of
    // the left-hand side.
    auto rhsWithExtraDimensions =
        other.getWithAtLeastNumDimensions(getNumOfResults());

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
    auto lhsWithExtraDimensions = this->getWithAtLeastNumDimensions(
        getNumOfDims() + rhsFakeDimensionsCount);

    llvm::SmallVector<std::unique_ptr<DimensionAccess>> lhsExtendedResults;

    for (const auto& result : getResults()) {
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

  IndexSet AccessFunction::map(const Point& point) const
  {
    IndexSet mappedIndices;

    for (const auto& result : getResults()) {
      mappedIndices = mappedIndices.append(
          result->map(point, getFakeDimensionsMap()));
    }

    return mappedIndices;
  }

  IndexSet AccessFunction::map(const IndexSet& indices) const
  {
    IndexSet mappedIndices;

    for (Point point : indices) {
      mappedIndices += map(point);
    }

    return mappedIndices;
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

  const DimensionAccess::FakeDimensionsMap&
  AccessFunction::getFakeDimensionsMap() const
  {
    return fakeDimensionsMap;
  }

  mlir::AffineMap AccessFunction::getExtendedAffineMap(
      DimensionAccess::FakeDimensionsMap& fakeDimensionsMap) const
  {
    llvm::SmallVector<mlir::AffineExpr> expressions;

    for (const auto& result : getResults()) {
      expressions.push_back(
          result->getAffineExpr(getNumOfDims(), fakeDimensionsMap));
    }

    return mlir::AffineMap::get(
        getNumOfDims() + fakeDimensionsMap.size(), 0,
        expressions, getContext());
  }

  std::unique_ptr<AccessFunction>
  AccessFunction::getWithAtLeastNumDimensions(unsigned int requestedDims) const
  {
    return AccessFunction::build(
        getContext(), std::max(numOfDimensions, requestedDims), getResults());
  }

  llvm::raw_ostream& operator<<(
      llvm::raw_ostream& os, const AccessFunction& obj)
  {
    return obj.dump(os);
  }
}
