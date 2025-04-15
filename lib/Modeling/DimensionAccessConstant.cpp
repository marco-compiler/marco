#include "marco/Modeling/DimensionAccessConstant.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling {
DimensionAccessConstant::DimensionAccessConstant(mlir::MLIRContext *context,
                                                 int64_t value)
    : DimensionAccess(DimensionAccess::Kind::Constant, context), value(value) {}

std::unique_ptr<DimensionAccess> DimensionAccessConstant::clone() const {
  return std::make_unique<DimensionAccessConstant>(*this);
}

bool DimensionAccessConstant::operator==(const DimensionAccess &other) const {
  if (auto otherCasted = other.dyn_cast<DimensionAccessConstant>()) {
    return *this == *otherCasted;
  }

  return false;
}

bool DimensionAccessConstant::operator==(
    const DimensionAccessConstant &other) const {
  return getValue() == other.getValue();
}

bool DimensionAccessConstant::operator!=(const DimensionAccess &other) const {
  if (auto otherCasted = other.dyn_cast<DimensionAccessConstant>()) {
    return *this != *otherCasted;
  }

  return true;
}

bool DimensionAccessConstant::operator!=(
    const DimensionAccessConstant &other) const {
  return getValue() != other.getValue();
}

llvm::raw_ostream &
DimensionAccessConstant::dump(llvm::raw_ostream &os,
                              const llvm::DenseMap<const IndexSet *, uint64_t>
                                  &iterationSpacesIds) const {
  return os << getValue();
}

void DimensionAccessConstant::collectIterationSpaces(
    llvm::DenseSet<const IndexSet *> &iterationSpaces) const {}

void DimensionAccessConstant::collectIterationSpaces(
    llvm::SmallVectorImpl<const IndexSet *> &iterationSpaces,
    llvm::DenseMap<const IndexSet *, llvm::DenseSet<uint64_t>>
        &dependentDimensions) const {}

bool DimensionAccessConstant::isConstant() const { return true; }

bool DimensionAccessConstant::isAffine() const { return true; }

mlir::AffineExpr DimensionAccessConstant::getAffineExpr() const {
  return mlir::getAffineConstantExpr(getValue(), getContext());
}

mlir::AffineExpr DimensionAccessConstant::getAffineExpr(
    unsigned int numOfDimensions,
    DimensionAccess::FakeDimensionsMap &fakeDimensionsMap) const {
  return mlir::getAffineConstantExpr(getValue(), getContext());
}

IndexSet DimensionAccessConstant::map(
    const Point &point,
    llvm::DenseMap<const IndexSet *, Point> &currentIndexSetsPoint) const {
  return {Point(getValue())};
}

int64_t DimensionAccessConstant::getValue() const { return value; }
} // namespace marco::modeling
