#include "marco/Modeling/DimensionAccessIndices.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  DimensionAccessIndices::DimensionAccessIndices(
      mlir::MLIRContext* context,
      std::shared_ptr<IndexSet> space,
      uint64_t dimension)
      : DimensionAccess(DimensionAccess::Kind::Indices, context),
        space(space),
        dimension(dimension)
  {
    assert(dimension < space->rank());
  }

  DimensionAccessIndices::DimensionAccessIndices(
      const DimensionAccessIndices& other) = default;

  DimensionAccessIndices::DimensionAccessIndices(
      DimensionAccessIndices&& other) noexcept = default;

  DimensionAccessIndices::~DimensionAccessIndices() = default;

  DimensionAccessIndices& DimensionAccessIndices::operator=(
      const DimensionAccessIndices& other)
  {
    DimensionAccessIndices result(other);
    swap(*this, result);
    return *this;
  }

  DimensionAccessIndices& DimensionAccessIndices::operator=(
      DimensionAccessIndices&& other) noexcept = default;

  void swap(DimensionAccessIndices& first, DimensionAccessIndices& second)
  {
    using std::swap;

    swap(static_cast<DimensionAccess&>(first),
         static_cast<DimensionAccess&>(second));

    swap(first.space, second.space);
    swap(first.dimension, second.dimension);
  }

  std::unique_ptr<DimensionAccess> DimensionAccessIndices::clone() const
  {
    return std::make_unique<DimensionAccessIndices>(*this);
  }

  bool DimensionAccessIndices::operator==(const DimensionAccess& other) const
  {
    if (auto otherCasted = other.dyn_cast<DimensionAccessIndices>()) {
      return *this == *otherCasted;
    }

    return false;
  }

  bool DimensionAccessIndices::operator==(
      const DimensionAccessIndices& other) const
  {
    return space == other.space && dimension == other.dimension;
  }

  bool DimensionAccessIndices::operator!=(const DimensionAccess& other) const
  {
    if (auto otherCasted = other.dyn_cast<DimensionAccessIndices>()) {
      return *this != *otherCasted;
    }

    return true;
  }

  bool DimensionAccessIndices::operator!=(
      const DimensionAccessIndices& other) const
  {
    return !(*this == other);
  }

  llvm::raw_ostream& DimensionAccessIndices::dump(
      llvm::raw_ostream& os,
      const llvm::DenseMap<IndexSet*, uint64_t>& indexSetsIds) const
  {
    auto it = indexSetsIds.find(space.get());
    assert(it != indexSetsIds.end());
    return os << "e" << it->getSecond() << "[" << dimension << "]";
  }

  void DimensionAccessIndices::collectIndexSets(
      llvm::SmallVectorImpl<IndexSet*>& indexSets) const
  {
    indexSets.push_back(space.get());
  }

  mlir::AffineExpr DimensionAccessIndices::getAffineExpr(
      unsigned int numOfDimensions,
      DimensionAccess::FakeDimensionsMap& fakeDimensionsMap) const
  {
    unsigned int numOfFakeDimensions = fakeDimensionsMap.size();

    fakeDimensionsMap[numOfDimensions + numOfFakeDimensions] =
        Redirect(clone());

    return mlir::getAffineDimExpr(
        numOfDimensions + numOfFakeDimensions, getContext());
  }

  IndexSet DimensionAccessIndices::map(const Point& point) const
  {
    return getIndices();
  }

  IndexSet& DimensionAccessIndices::getIndices()
  {
    return *space;
  }

  const IndexSet& DimensionAccessIndices::getIndices() const
  {
    return *space;
  }
}
