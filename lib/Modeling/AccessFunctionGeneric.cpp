#include "marco/Modeling/AccessFunctionGeneric.h"
#include "marco/Modeling/DimensionAccessConstant.h"
#include "marco/Modeling/DimensionAccessDimension.h"
#include "llvm/Support/raw_os_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  AccessFunctionGeneric::AccessFunctionGeneric(
      mlir::MLIRContext* context,
      uint64_t numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results,
      DimensionAccess::FakeDimensionsMap fakeDimensionsMap)
      : AccessFunction(AccessFunction::Kind::Generic, context),
        numOfDimensions(numOfDimensions),
        fakeDimensionsMap(std::move(fakeDimensionsMap))
  {
    for (const auto& result : results) {
      this->results.push_back(result->clone());
    }
  }

  AccessFunctionGeneric::AccessFunctionGeneric(
      Kind kind,
      mlir::MLIRContext* context,
      uint64_t numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results,
      DimensionAccess::FakeDimensionsMap fakeDimensionsMap)
      : AccessFunction(kind, context),
        numOfDimensions(numOfDimensions),
        fakeDimensionsMap(std::move(fakeDimensionsMap))
  {
    for (const auto& result : results) {
      this->results.push_back(result->clone());
    }
  }

  AccessFunctionGeneric::AccessFunctionGeneric(mlir::AffineMap affineMap)
      : AccessFunctionGeneric(
          affineMap.getContext(),
          affineMap.getNumDims(),
          convertAffineExpressions(affineMap.getResults()),
          DimensionAccess::FakeDimensionsMap())
  {
  }

  AccessFunctionGeneric::AccessFunctionGeneric(const AccessFunctionGeneric& other)
      : AccessFunction(other),
        numOfDimensions(other.numOfDimensions),
        fakeDimensionsMap(other.fakeDimensionsMap)
  {
    for (const auto& result : other.results) {
      results.push_back(result->clone());
    }
  }

  AccessFunctionGeneric::AccessFunctionGeneric(
      AccessFunctionGeneric&& other) = default;

  AccessFunctionGeneric::~AccessFunctionGeneric() = default;

  AccessFunctionGeneric& AccessFunctionGeneric::operator=(
      const AccessFunctionGeneric& other)
  {
    AccessFunctionGeneric result(other);
    swap(*this, result);
    return *this;
  }

  AccessFunctionGeneric& AccessFunctionGeneric::operator=(
      AccessFunctionGeneric&& other) = default;

  void swap(AccessFunctionGeneric& first, AccessFunctionGeneric& second)
  {
    using std::swap;

    swap(static_cast<AccessFunction&>(first),
         static_cast<AccessFunction&>(second));

    swap(first.numOfDimensions, second.numOfDimensions);

    llvm::SmallVector<std::unique_ptr<DimensionAccess>> firstTmp =
        std::move(first.results);

    first.results = std::move(second.results);
    second.results = std::move(firstTmp);

    swap(first.fakeDimensionsMap, second.fakeDimensionsMap);
  }

  std::unique_ptr<AccessFunction> AccessFunctionGeneric::clone() const
  {
    return std::make_unique<AccessFunctionGeneric>(*this);
  }

  bool AccessFunctionGeneric::operator==(const AccessFunction& other) const
  {
    if (auto otherCasted = other.dyn_cast<AccessFunctionGeneric>()) {
      return *this == *otherCasted;
    }

    return false;
  }

  bool AccessFunctionGeneric::operator==(
      const AccessFunctionGeneric& other) const
  {
    if (numOfDimensions != other.numOfDimensions) {
      return false;
    }

    for (const auto& [lhsResult, rhsResult] :
         llvm::zip(getResults(), other.getResults())) {
      if (*lhsResult != *rhsResult) {
        return false;
      }
    }

    return fakeDimensionsMap == other.fakeDimensionsMap;
  }

  bool AccessFunctionGeneric::operator!=(const AccessFunction& other) const
  {
    return !(*this == other);
  }


  bool AccessFunctionGeneric::operator!=(
      const AccessFunctionGeneric& other) const
  {
    return !(*this == other);
  }

  llvm::raw_ostream& AccessFunctionGeneric::dump(llvm::raw_ostream& os) const
  {
    os << "(";

    for (size_t i = 0, e = getNumOfDims(); i < e; ++i) {
      if (i != 0) {
        os << ", ";
      }

      os << "d" << i;
    }

    os << ") -> (";

    llvm::SmallVector<IndexSet*> indexSets;

    for (const auto& result : getResults()) {
      result->collectIndexSets(indexSets);
    }

    llvm::DenseMap<IndexSet*, uint64_t> indexSetsIds;

    for (size_t i = 0, e = indexSets.size(); i < e; ++i) {
      indexSetsIds[indexSets[i]] = indexSets.size();
    }

    for (size_t i = 0, e = getNumOfResults(); i < e; ++i) {
      if (i != 0) {
        os << ", ";
      }

      getResults()[i]->dump(os, indexSetsIds);
    }

    os << ")";

    if (!indexSets.empty()) {
      for (size_t i = 0, e = indexSets.size(); i < e; ++i) {
        os << ", e" << i << ": " << *indexSets[i];
      }
    }

    return os;
  }

  uint64_t AccessFunctionGeneric::getNumOfDims() const
  {
    return numOfDimensions;
  }

  uint64_t AccessFunctionGeneric::getNumOfResults() const
  {
    return results.size();
  }

  bool AccessFunctionGeneric::isAffine() const
  {
    return llvm::all_of(
        getResults(),
        [](const std::unique_ptr<DimensionAccess>& result) {
          return result->isAffine();
        });
  }

  mlir::AffineMap AccessFunctionGeneric::getAffineMap() const
  {
    assert(isAffine());
    llvm::SmallVector<mlir::AffineExpr> expressions;

    for (const auto& result : getResults()) {
      expressions.push_back(result->getAffineExpr());
    }

    return mlir::AffineMap::get(getNumOfDims(), 0, expressions, getContext());
  }

  bool AccessFunctionGeneric::isIdentity() const
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

  IndexSet AccessFunctionGeneric::map(const Point& point) const
  {
    IndexSet mappedIndices;

    for (const auto& result : getResults()) {
      mappedIndices = mappedIndices.append(
          result->map(point, fakeDimensionsMap));
    }

    return mappedIndices;
  }

  IndexSet AccessFunctionGeneric::map(const IndexSet& indices) const
  {
    IndexSet mappedIndices;

    for (Point point : indices) {
      mappedIndices += map(point);
    }

    return mappedIndices;
  }

  llvm::SmallVector<std::unique_ptr<DimensionAccess>, 6>
  AccessFunctionGeneric::getGeneralizedAccesses() const
  {
    llvm::SmallVector<std::unique_ptr<DimensionAccess>, 6> dimensionAccesses;

    for (const auto& result : getResults()) {
      dimensionAccesses.push_back(result->clone());
    }

    return dimensionAccesses;
  }

  mlir::AffineMap AccessFunctionGeneric::getExtendedAffineMap(
      DimensionAccess::FakeDimensionsMap& map) const
  {
    llvm::SmallVector<mlir::AffineExpr> expressions;

    for (const auto& result : getResults()) {
      expressions.push_back(result->getAffineExpr(getNumOfDims(), map));
    }

    return mlir::AffineMap::get(
        getNumOfDims() + map.size(), 0, expressions, getContext());
  }

  llvm::ArrayRef<std::unique_ptr<DimensionAccess>>
  AccessFunctionGeneric::getResults() const
  {
    return results;
  }
}
