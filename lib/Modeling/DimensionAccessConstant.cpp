#include "marco/Modeling/DimensionAccessConstant.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  DimensionAccessConstant::DimensionAccessConstant(
      mlir::MLIRContext* context, int64_t value)
      : DimensionAccess(DimensionAccess::Kind::Constant, context),
        value(value)
  {
  }

  DimensionAccessConstant::DimensionAccessConstant(
      const DimensionAccessConstant& other)
      : DimensionAccess(other),
        value(other.value)
  {
  }

  DimensionAccessConstant::DimensionAccessConstant(
      DimensionAccessConstant&& other) noexcept = default;

  DimensionAccessConstant::~DimensionAccessConstant() = default;

  DimensionAccessConstant& DimensionAccessConstant::operator=(
      const DimensionAccessConstant& other)
  {
    DimensionAccessConstant result(other);
    swap(*this, result);
    return *this;
  }

  DimensionAccessConstant& DimensionAccessConstant::operator=(
      DimensionAccessConstant&& other) noexcept = default;

  void swap(DimensionAccessConstant& first, DimensionAccessConstant& second)
  {
    using std::swap;

    swap(static_cast<DimensionAccess&>(first),
         static_cast<DimensionAccess&>(second));

    swap(first.value, second.value);
  }

  std::unique_ptr<DimensionAccess> DimensionAccessConstant::clone() const
  {
    return std::make_unique<DimensionAccessConstant>(*this);
  }

  bool DimensionAccessConstant::operator==(const DimensionAccess& other) const
  {
    if (auto otherCasted = other.dyn_cast<DimensionAccessConstant>()) {
      return *this == *otherCasted;
    }

    return false;
  }

  bool DimensionAccessConstant::operator==(
      const DimensionAccessConstant& other) const
  {
    return getValue() == other.getValue();
  }

  bool DimensionAccessConstant::operator!=(const DimensionAccess& other) const
  {
    if (auto otherCasted = other.dyn_cast<DimensionAccessConstant>()) {
      return *this != *otherCasted;
    }

    return true;
  }

  bool DimensionAccessConstant::operator!=(
      const DimensionAccessConstant& other) const
  {
    return getValue() != other.getValue();
  }

  llvm::raw_ostream& DimensionAccessConstant::dump(
      llvm::raw_ostream& os,
      const llvm::DenseMap<IndexSet*, uint64_t>& indexSetsIds) const
  {
    return os << getValue();
  }

  void DimensionAccessConstant::collectIndexSets(
      llvm::SmallVectorImpl<IndexSet*>& indexSets) const
  {
  }

  bool DimensionAccessConstant::isAffine() const
  {
    return true;
  }

  mlir::AffineExpr DimensionAccessConstant::getAffineExpr() const
  {
    return mlir::getAffineConstantExpr(getValue(), getContext());
  }

  mlir::AffineExpr DimensionAccessConstant::getAffineExpr(
      unsigned int numOfDimensions,
      DimensionAccess::FakeDimensionsMap& fakeDimensionsMap) const
  {
    return mlir::getAffineConstantExpr(getValue(), getContext());
  }

  IndexSet DimensionAccessConstant::map(const Point& point) const
  {
    return {Point(getValue())};
  }

  int64_t DimensionAccessConstant::getValue() const
  {
    return value;
  }
}
