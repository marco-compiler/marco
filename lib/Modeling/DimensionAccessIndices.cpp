#include "marco/Modeling/DimensionAccessIndices.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling {
DimensionAccessIndices::DimensionAccessIndices(
    mlir::MLIRContext *context, std::shared_ptr<IndexSet> space,
    uint64_t dimension, llvm::SetVector<uint64_t> dimensionDependencies)
    : DimensionAccess(DimensionAccess::Kind::Indices, context), space(space),
      dimension(dimension),
      dimensionDependencies(std::move(dimensionDependencies)) {
  assert(dimension < space->rank());
}

std::unique_ptr<DimensionAccess> DimensionAccessIndices::clone() const {
  return std::make_unique<DimensionAccessIndices>(*this);
}

bool DimensionAccessIndices::operator==(const DimensionAccess &other) const {
  if (auto otherCasted = other.dyn_cast<DimensionAccessIndices>()) {
    return *this == *otherCasted;
  }

  return false;
}

bool DimensionAccessIndices::operator==(
    const DimensionAccessIndices &other) const {
  return space == other.space && dimension == other.dimension;
}

bool DimensionAccessIndices::operator!=(const DimensionAccess &other) const {
  if (auto otherCasted = other.dyn_cast<DimensionAccessIndices>()) {
    return *this != *otherCasted;
  }

  return true;
}

bool DimensionAccessIndices::operator!=(
    const DimensionAccessIndices &other) const {
  return !(*this == other);
}

llvm::raw_ostream &
DimensionAccessIndices::dump(llvm::raw_ostream &os,
                             const llvm::DenseMap<const IndexSet *, uint64_t>
                                 &iterationSpacesIds) const {
  auto it = iterationSpacesIds.find(space.get());
  assert(it != iterationSpacesIds.end());
  return os << "e" << it->getSecond() << "[" << dimension << "]";
}

void DimensionAccessIndices::collectIterationSpaces(
    llvm::SetVector<const IndexSet *> &iterationSpaces) const {
  iterationSpaces.insert(space.get());
}

void DimensionAccessIndices::collectIterationSpaces(
    llvm::SmallVectorImpl<const IndexSet *> &iterationSpaces,
    llvm::DenseMap<const IndexSet *, llvm::SetVector<uint64_t>>
        &dependentDimensions) const {
  iterationSpaces.push_back(space.get());

  if (!dimensionDependencies.empty()) {
    dependentDimensions[space.get()].insert(dimension);

    dependentDimensions[space.get()].insert(dimensionDependencies.begin(),
                                            dimensionDependencies.end());
  }
}

bool DimensionAccessIndices::isConstant() const { return true; }

mlir::AffineExpr DimensionAccessIndices::getAffineExpr(
    unsigned int numOfDimensions,
    DimensionAccess::FakeDimensionsMap &fakeDimensionsMap) const {
  unsigned int numOfFakeDimensions = fakeDimensionsMap.size();

  fakeDimensionsMap[numOfDimensions + numOfFakeDimensions] = Redirect(clone());

  return mlir::getAffineDimExpr(numOfDimensions + numOfFakeDimensions,
                                getContext());
}

IndexSet DimensionAccessIndices::map(
    const Point &point,
    llvm::DenseMap<const IndexSet *, Point> &currentIndexSetsPoint) const {
  IndexSet allIndices = getIndices();

  if (dimensionDependencies.empty()) {
    IndexSet result;

    for (const MultidimensionalRange &range :
         llvm::make_range(allIndices.rangesBegin(), allIndices.rangesEnd())) {
      result += MultidimensionalRange(range[dimension]);
    }

    return result;
  }

  auto pointIt = currentIndexSetsPoint.find(space.get());
  assert(pointIt != currentIndexSetsPoint.end());
  return {Point(pointIt->getSecond()[dimension])};
}

IndexSet &DimensionAccessIndices::getIndices() { return *space; }

const IndexSet &DimensionAccessIndices::getIndices() const { return *space; }
} // namespace marco::modeling
