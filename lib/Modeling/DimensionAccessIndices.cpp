#include "marco/Modeling/DimensionAccessIndices.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  DimensionAccessIndices::DimensionAccessIndices(
      mlir::MLIRContext* context, IndexSet indices)
      : DimensionAccess(DimensionAccess::Kind::Indices, context),
        resultIndices(std::move(indices))
  {
    assert(getIndices().rank() == 1);
  }

  DimensionAccessIndices::DimensionAccessIndices(
      const DimensionAccessIndices& other)
      : DimensionAccess(other),
        resultIndices(other.resultIndices)
  {
  }

  DimensionAccessIndices::DimensionAccessIndices(
      DimensionAccessIndices&& other) = default;

  DimensionAccessIndices::~DimensionAccessIndices() = default;

  DimensionAccessIndices& DimensionAccessIndices::operator=(
      const DimensionAccessIndices& other)
  {
    DimensionAccessIndices result(other);
    swap(*this, result);
    return *this;
  }

  DimensionAccessIndices& DimensionAccessIndices::operator=(
      DimensionAccessIndices&& other) = default;

  void swap(DimensionAccessIndices& first, DimensionAccessIndices& second)
  {
    using std::swap;

    swap(static_cast<DimensionAccess&>(first),
         static_cast<DimensionAccess&>(second));

    swap(first.resultIndices, second.resultIndices);
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

  bool DimensionAccessIndices::operator==(const DimensionAccessIndices& other) const
  {
    return getIndices() == other.getIndices();
  }

  bool DimensionAccessIndices::operator!=(const DimensionAccess& other) const
  {
    if (auto otherCasted = other.dyn_cast<DimensionAccessIndices>()) {
      return *this != *otherCasted;
    }

    return true;
  }

  bool DimensionAccessIndices::operator!=(const DimensionAccessIndices& other) const
  {
    return getIndices() != other.getIndices();
  }

  llvm::raw_ostream& DimensionAccessIndices::dump(llvm::raw_ostream& os) const
  {
    return os << getIndices();
  }

  mlir::AffineExpr DimensionAccessIndices::getAffineExpr(
      unsigned int numOfDimensions,
      DimensionAccess::FakeDimensionsMap& fakeDimensionsMap) const
  {
    unsigned int numOfFakeDimensions = fakeDimensionsMap.size();
    fakeDimensionsMap[numOfDimensions + numOfFakeDimensions] = this;

    return mlir::getAffineDimExpr(
        numOfDimensions + numOfFakeDimensions, getContext());
  }

  IndexSet DimensionAccessIndices::map(const Point& point) const
  {
    return getIndices();
  }

  IndexSet& DimensionAccessIndices::getIndices()
  {
    return resultIndices;
  }

  const IndexSet& DimensionAccessIndices::getIndices() const
  {
    return resultIndices;
  }
}
