#include "marco/Modeling/DimensionAccessRange.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling {
DimensionAccessRange::DimensionAccessRange(mlir::MLIRContext *context,
                                           Range range)
    : DimensionAccess(DimensionAccess::Kind::Range, context), range(range) {}

DimensionAccessRange::DimensionAccessRange(const DimensionAccessRange &other) =
    default;

DimensionAccessRange::DimensionAccessRange(
    DimensionAccessRange &&other) noexcept = default;

DimensionAccessRange::~DimensionAccessRange() = default;

DimensionAccessRange &
DimensionAccessRange::operator=(const DimensionAccessRange &other) {
  DimensionAccessRange result(other);
  swap(*this, result);
  return *this;
}

DimensionAccessRange &DimensionAccessRange::operator=(
    DimensionAccessRange &&other) noexcept = default;

void swap(DimensionAccessRange &first, DimensionAccessRange &second) {
  using std::swap;

  swap(static_cast<DimensionAccess &>(first),
       static_cast<DimensionAccess &>(second));

  swap(first.range, second.range);
}

std::unique_ptr<DimensionAccess> DimensionAccessRange::clone() const {
  return std::make_unique<DimensionAccessRange>(*this);
}

bool DimensionAccessRange::operator==(const DimensionAccess &other) const {
  if (auto otherCasted = other.dyn_cast<DimensionAccessRange>()) {
    return *this == *otherCasted;
  }

  return false;
}

bool DimensionAccessRange::operator==(const DimensionAccessRange &other) const {
  return getRange() == other.getRange();
}

bool DimensionAccessRange::operator!=(const DimensionAccess &other) const {
  if (auto otherCasted = other.dyn_cast<DimensionAccessRange>()) {
    return *this != *otherCasted;
  }

  return true;
}

bool DimensionAccessRange::operator!=(const DimensionAccessRange &other) const {
  return getRange() != other.getRange();
}

llvm::raw_ostream &
DimensionAccessRange::dump(llvm::raw_ostream &os,
                           const llvm::DenseMap<const IndexSet *, uint64_t>
                               &iterationSpacesIds) const {
  return os << getRange();
}

void DimensionAccessRange::collectIterationSpaces(
    llvm::DenseSet<const IndexSet *> &iterationSpaces) const {}

void DimensionAccessRange::collectIterationSpaces(
    llvm::SmallVectorImpl<const IndexSet *> &iterationSpaces,
    llvm::DenseMap<const IndexSet *, llvm::DenseSet<uint64_t>>
        &dependentDimensions) const {}

mlir::AffineExpr DimensionAccessRange::getAffineExpr(
    unsigned int numOfDimensions,
    DimensionAccess::FakeDimensionsMap &fakeDimensionsMap) const {
  unsigned int numOfFakeDimensions = fakeDimensionsMap.size();

  fakeDimensionsMap[numOfDimensions + numOfFakeDimensions] = Redirect(clone());

  return mlir::getAffineDimExpr(numOfDimensions + numOfFakeDimensions,
                                getContext());
}

IndexSet DimensionAccessRange::map(
    const Point &point,
    llvm::DenseMap<const IndexSet *, Point> &currentIndexSetsPoint) const {
  return IndexSet{MultidimensionalRange(getRange())};
}

Range &DimensionAccessRange::getRange() { return range; }

const Range &DimensionAccessRange::getRange() const { return range; }
} // namespace marco::modeling
