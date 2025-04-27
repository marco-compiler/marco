#include "marco/Modeling/DimensionAccessDimension.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling {
DimensionAccessDimension::DimensionAccessDimension(mlir::MLIRContext *context,
                                                   uint64_t dimension)
    : DimensionAccess(DimensionAccess::Kind::Dimension, context),
      dimension(dimension) {}

std::unique_ptr<DimensionAccess> DimensionAccessDimension::clone() const {
  return std::make_unique<DimensionAccessDimension>(*this);
}

bool DimensionAccessDimension::operator==(const DimensionAccess &other) const {
  if (auto otherCasted = other.dyn_cast<DimensionAccessDimension>()) {
    return *this == *otherCasted;
  }

  return false;
}

bool DimensionAccessDimension::operator==(
    const DimensionAccessDimension &other) const {
  return getDimension() == other.getDimension();
}

bool DimensionAccessDimension::operator!=(const DimensionAccess &other) const {
  if (auto otherCasted = other.dyn_cast<DimensionAccessDimension>()) {
    return *this != *otherCasted;
  }

  return true;
}

llvm::raw_ostream &
DimensionAccessDimension::dump(llvm::raw_ostream &os,
                               const llvm::DenseMap<const IndexSet *, uint64_t>
                                   &iterationSpacesIds) const {
  return os << "d" << getDimension();
}

bool DimensionAccessDimension::operator!=(
    const DimensionAccessDimension &other) const {
  return getDimension() != other.getDimension();
}

void DimensionAccessDimension::collectIterationSpaces(
    llvm::SetVector<const IndexSet *> &iterationSpaces) const {}

void DimensionAccessDimension::collectIterationSpaces(
    llvm::SmallVectorImpl<const IndexSet *> &iterationSpaces,
    llvm::DenseMap<const IndexSet *, llvm::SetVector<uint64_t>>
        &dependentDimensions) const {}

bool DimensionAccessDimension::isConstant() const { return false; }

bool DimensionAccessDimension::isAffine() const { return true; }

mlir::AffineExpr DimensionAccessDimension::getAffineExpr() const {
  return mlir::getAffineDimExpr(getDimension(), getContext());
}

mlir::AffineExpr DimensionAccessDimension::getAffineExpr(
    unsigned int numOfDimensions,
    DimensionAccess::FakeDimensionsMap &fakeDimensionsMap) const {
  return mlir::getAffineDimExpr(getDimension(), getContext());
}

IndexSet DimensionAccessDimension::map(
    const Point &point,
    llvm::DenseMap<const IndexSet *, Point> &currentIndexSetsPoint) const {
  return {Point(point[getDimension()])};
}

uint64_t DimensionAccessDimension::getDimension() const { return dimension; }
} // namespace marco::modeling
