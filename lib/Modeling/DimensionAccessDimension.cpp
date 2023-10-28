#include "marco/Modeling/DimensionAccessDimension.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  DimensionAccessDimension::DimensionAccessDimension(
      mlir::MLIRContext* context, uint64_t dimension)
      : DimensionAccess(DimensionAccess::Kind::Dimension, context),
        dimension(dimension)
  {
  }

  DimensionAccessDimension::DimensionAccessDimension(
      const DimensionAccessDimension& other)
      : DimensionAccess(other),
        dimension(other.dimension)
  {
  }

  DimensionAccessDimension::DimensionAccessDimension(
      DimensionAccessDimension&& other) = default;

  DimensionAccessDimension::~DimensionAccessDimension() = default;

  DimensionAccessDimension& DimensionAccessDimension::operator=(
      const DimensionAccessDimension& other)
  {
    DimensionAccessDimension result(other);
    swap(*this, result);
    return *this;
  }

  DimensionAccessDimension& DimensionAccessDimension::operator=(
      DimensionAccessDimension&& other) = default;

  void swap(DimensionAccessDimension& first, DimensionAccessDimension& second)
  {
    using std::swap;

    swap(static_cast<DimensionAccess&>(first),
         static_cast<DimensionAccess&>(second));

    swap(first.dimension, second.dimension);
  }

  std::unique_ptr<DimensionAccess> DimensionAccessDimension::clone() const
  {
    return std::make_unique<DimensionAccessDimension>(*this);
  }

  bool DimensionAccessDimension::operator==(const DimensionAccess& other) const
  {
    if (auto otherCasted = other.dyn_cast<DimensionAccessDimension>()) {
      return *this == *otherCasted;
    }

    return false;
  }

  bool DimensionAccessDimension::operator==(const DimensionAccessDimension& other) const
  {
    return getDimension() == other.getDimension();
  }

  bool DimensionAccessDimension::operator!=(const DimensionAccess& other) const
  {
    if (auto otherCasted = other.dyn_cast<DimensionAccessDimension>()) {
      return *this != *otherCasted;
    }

    return true;
  }

  llvm::raw_ostream& DimensionAccessDimension::dump(
      llvm::raw_ostream& os) const
  {
    return os << "d" << getDimension();
  }

  bool DimensionAccessDimension::operator!=(const DimensionAccessDimension& other) const
  {
    return getDimension() != other.getDimension();
  }

  bool DimensionAccessDimension::isAffine() const
  {
    return true;
  }

  mlir::AffineExpr DimensionAccessDimension::getAffineExpr() const
  {
    return mlir::getAffineDimExpr(getDimension(), getContext());
  }

  mlir::AffineExpr DimensionAccessDimension::getAffineExpr(
      unsigned int numOfDimensions,
      DimensionAccess::FakeDimensionsMap& fakeDimensionsMap) const
  {
    return mlir::getAffineDimExpr(getDimension(), getContext());
  }

  IndexSet DimensionAccessDimension::map(const Point& point) const
  {
    return {Point(point[getDimension()])};
  }

  uint64_t DimensionAccessDimension::getDimension() const
  {
    return dimension;
  }
}
