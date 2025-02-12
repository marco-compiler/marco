#include "marco/Modeling/AccessFunctionGeneric.h"
#include "marco/Modeling/DimensionAccessConstant.h"
#include "marco/Modeling/DimensionAccessDimension.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_os_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling {
AccessFunctionGeneric::AccessFunctionGeneric(
    mlir::MLIRContext *context, uint64_t numOfDimensions,
    llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
    : AccessFunction(AccessFunction::Kind::Generic, context),
      numOfDimensions(numOfDimensions) {
  for (const auto &result : results) {
    this->results.push_back(result->clone());
  }
}

AccessFunctionGeneric::AccessFunctionGeneric(
    Kind kind, mlir::MLIRContext *context, uint64_t numOfDimensions,
    llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
    : AccessFunction(kind, context), numOfDimensions(numOfDimensions) {
  for (const auto &result : results) {
    this->results.push_back(result->clone());
  }
}

AccessFunctionGeneric::AccessFunctionGeneric(mlir::AffineMap affineMap)
    : AccessFunctionGeneric(affineMap.getContext(), affineMap.getNumDims(),
                            convertAffineExpressions(affineMap.getResults())) {}

AccessFunctionGeneric::AccessFunctionGeneric(const AccessFunctionGeneric &other)
    : AccessFunction(other), numOfDimensions(other.numOfDimensions) {
  for (const auto &result : other.results) {
    results.push_back(result->clone());
  }
}

AccessFunctionGeneric::AccessFunctionGeneric(AccessFunctionGeneric &&other) =
    default;

AccessFunctionGeneric::~AccessFunctionGeneric() = default;

AccessFunctionGeneric &
AccessFunctionGeneric::operator=(const AccessFunctionGeneric &other) {
  AccessFunctionGeneric result(other);
  swap(*this, result);
  return *this;
}

AccessFunctionGeneric &
AccessFunctionGeneric::operator=(AccessFunctionGeneric &&other) = default;

void swap(AccessFunctionGeneric &first, AccessFunctionGeneric &second) {
  using std::swap;

  swap(static_cast<AccessFunction &>(first),
       static_cast<AccessFunction &>(second));

  swap(first.numOfDimensions, second.numOfDimensions);

  llvm::SmallVector<std::unique_ptr<DimensionAccess>> firstTmp =
      std::move(first.results);

  first.results = std::move(second.results);
  second.results = std::move(firstTmp);
}

std::unique_ptr<AccessFunction> AccessFunctionGeneric::clone() const {
  return std::make_unique<AccessFunctionGeneric>(*this);
}

bool AccessFunctionGeneric::operator==(const AccessFunction &other) const {
  if (auto otherCasted = other.dyn_cast<AccessFunctionGeneric>()) {
    return *this == *otherCasted;
  }

  return false;
}

bool AccessFunctionGeneric::operator==(
    const AccessFunctionGeneric &other) const {
  if (numOfDimensions != other.numOfDimensions) {
    return false;
  }

  for (const auto &[lhsResult, rhsResult] :
       llvm::zip(getResults(), other.getResults())) {
    if (*lhsResult != *rhsResult) {
      return false;
    }
  }

  return true;
}

bool AccessFunctionGeneric::operator!=(const AccessFunction &other) const {
  return !(*this == other);
}

bool AccessFunctionGeneric::operator!=(
    const AccessFunctionGeneric &other) const {
  return !(*this == other);
}

llvm::raw_ostream &AccessFunctionGeneric::dump(llvm::raw_ostream &os) const {
  os << "(";

  for (size_t i = 0, e = getNumOfDims(); i < e; ++i) {
    if (i != 0) {
      os << ", ";
    }

    os << "d" << i;
  }

  os << ") -> (";

  llvm::DenseSet<const IndexSet *> iterationSpaces;

  for (const auto &result : getResults()) {
    result->collectIterationSpaces(iterationSpaces);
  }

  llvm::DenseMap<const IndexSet *, uint64_t> iterationSpacesIds;
  uint64_t currentIterationSpaceId = 0;

  for (const IndexSet *iterationSpace : iterationSpaces) {
    iterationSpacesIds[iterationSpace] = currentIterationSpaceId++;
  }

  for (size_t i = 0, e = getNumOfResults(); i < e; ++i) {
    if (i != 0) {
      os << ", ";
    }

    getResults()[i]->dump(os, iterationSpacesIds);
  }

  os << ")";

  if (!iterationSpacesIds.empty()) {
    for (const IndexSet *iterationSpace : iterationSpaces) {
      os << ", e" << iterationSpacesIds[iterationSpace] << ":"
         << *iterationSpace;
    }
  }

  return os;
}

uint64_t AccessFunctionGeneric::getNumOfDims() const { return numOfDimensions; }

uint64_t AccessFunctionGeneric::getNumOfResults() const {
  return results.size();
}

bool AccessFunctionGeneric::isAffine() const {
  return llvm::all_of(getResults(),
                      [](const std::unique_ptr<DimensionAccess> &result) {
                        return result->isAffine();
                      });
}

mlir::AffineMap AccessFunctionGeneric::getAffineMap() const {
  assert(isAffine());
  llvm::SmallVector<mlir::AffineExpr> expressions;

  for (const auto &result : getResults()) {
    expressions.push_back(result->getAffineExpr());
  }

  return mlir::AffineMap::get(getNumOfDims(), 0, expressions, getContext());
}

bool AccessFunctionGeneric::isIdentity() const {
  unsigned int dimension = 0;

  for (const auto &result : getResults()) {
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

IndexSet AccessFunctionGeneric::map(const Point &point) const {
  IndexSet mappedIndices;
  llvm::DenseMap<const IndexSet *, Point> currentIterationSpacePoint;

  // Get the iteration spaces and the dependencies among their dimensions.
  llvm::SmallVector<const IndexSet *, 10> iterationSpaces;

  llvm::DenseMap<const IndexSet *, llvm::DenseSet<uint64_t>>
      iterationSpacesDependencies;

  collectIterationSpaces(iterationSpaces, iterationSpacesDependencies);

  // Get the unique iteration spaces.
  llvm::DenseSet<const IndexSet *> uniqueIterationSpaces(
      iterationSpaces.begin(), iterationSpaces.end());

  // Determine the iteration spaces with dependencies.
  llvm::SmallVector<const IndexSet *, 10> iterationSpacesWithDependencies;

  for (const IndexSet *iterationSpace : uniqueIterationSpaces) {
    if (!iterationSpacesDependencies[iterationSpace].empty()) {
      iterationSpacesWithDependencies.push_back(iterationSpace);
    }
  }

  if (!iterationSpacesWithDependencies.empty()) {
    map(mappedIndices, point, iterationSpacesWithDependencies, 0,
        currentIterationSpacePoint);
  }

  // No dependencies among iteration spaces.
  for (const auto &result : getResults()) {
    mappedIndices =
        mappedIndices.append(result->map(point, currentIterationSpacePoint));
  }

  return mappedIndices;
}

void AccessFunctionGeneric::map(
    IndexSet &mappedIndices, const Point &point,
    llvm::ArrayRef<const IndexSet *> iterationSpaces,
    size_t currentIterationSpace,
    llvm::DenseMap<const IndexSet *, Point> &currentIterationSpacePoint) const {
  if (currentIterationSpace < iterationSpaces.size()) {
    const IndexSet *iterationSpace = iterationSpaces[currentIterationSpace];

    for (Point iterationSpacePoint : *iterationSpace) {
      currentIterationSpacePoint[iterationSpace] = iterationSpacePoint;

      map(mappedIndices, point, iterationSpaces, currentIterationSpace + 1,
          currentIterationSpacePoint);
    }
  }

  IndexSet currentMappedIndices;

  for (const auto &result : getResults()) {
    currentMappedIndices = currentMappedIndices.append(
        result->map(point, currentIterationSpacePoint));
  }

  mappedIndices += currentMappedIndices;
}

IndexSet AccessFunctionGeneric::map(const IndexSet &indices) const {
  IndexSet mappedIndices;

  for (Point point : indices) {
    mappedIndices += map(point);
  }

  return mappedIndices;
}

llvm::SmallVector<std::unique_ptr<DimensionAccess>, 6>
AccessFunctionGeneric::getGeneralizedAccesses() const {
  llvm::SmallVector<std::unique_ptr<DimensionAccess>, 6> dimensionAccesses;

  for (const auto &result : getResults()) {
    dimensionAccesses.push_back(result->clone());
  }

  return dimensionAccesses;
}

mlir::AffineMap AccessFunctionGeneric::getExtendedAffineMap(
    DimensionAccess::FakeDimensionsMap &map) const {
  llvm::SmallVector<mlir::AffineExpr> expressions;

  for (const auto &result : getResults()) {
    expressions.push_back(result->getAffineExpr(getNumOfDims(), map));
  }

  return mlir::AffineMap::get(getNumOfDims() + map.size(), 0, expressions,
                              getContext());
}

llvm::ArrayRef<std::unique_ptr<DimensionAccess>>
AccessFunctionGeneric::getResults() const {
  return results;
}

void AccessFunctionGeneric::collectIterationSpaces(
    llvm::SmallVectorImpl<const IndexSet *> &iterationSpaces,
    llvm::DenseMap<const IndexSet *, llvm::DenseSet<uint64_t>>
        &dependendentDimensions) const {
  for (const auto &result : getResults()) {
    result->collectIterationSpaces(iterationSpaces, dependendentDimensions);
  }
}
} // namespace marco::modeling
