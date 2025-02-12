#include "marco/Modeling/DimensionAccessAdd.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling {
DimensionAccessAdd::DimensionAccessAdd(mlir::MLIRContext *context,
                                       std::unique_ptr<DimensionAccess> first,
                                       std::unique_ptr<DimensionAccess> second)
    : DimensionAccess(DimensionAccess::Kind::Add, context),
      first(std::move(first)), second(std::move(second)) {}

DimensionAccessAdd::DimensionAccessAdd(const DimensionAccessAdd &other)
    : DimensionAccess(other), first(other.getFirst().clone()),
      second(other.getSecond().clone()) {}

DimensionAccessAdd::DimensionAccessAdd(DimensionAccessAdd &&other) noexcept =
    default;

DimensionAccessAdd::~DimensionAccessAdd() = default;

DimensionAccessAdd &
DimensionAccessAdd::operator=(const DimensionAccessAdd &other) {
  DimensionAccessAdd result(other);
  swap(*this, result);
  return *this;
}

DimensionAccessAdd &
DimensionAccessAdd::operator=(DimensionAccessAdd &&other) noexcept = default;

void swap(DimensionAccessAdd &first, DimensionAccessAdd &second) {
  using std::swap;

  swap(static_cast<DimensionAccess &>(first),
       static_cast<DimensionAccess &>(second));

  swap(first.first, second.first);
  swap(first.second, second.second);
}

std::unique_ptr<DimensionAccess> DimensionAccessAdd::clone() const {
  return std::make_unique<DimensionAccessAdd>(*this);
}

bool DimensionAccessAdd::operator==(const DimensionAccess &other) const {
  if (auto otherCasted = other.dyn_cast<DimensionAccessAdd>()) {
    return *this == *otherCasted;
  }

  return false;
}

bool DimensionAccessAdd::operator==(const DimensionAccessAdd &other) const {
  return getFirst() == other.getFirst() && getSecond() == other.getSecond();
}

bool DimensionAccessAdd::operator!=(const DimensionAccess &other) const {
  if (auto otherCasted = other.dyn_cast<DimensionAccessAdd>()) {
    return *this != *otherCasted;
  }

  return true;
}

bool DimensionAccessAdd::operator!=(const DimensionAccessAdd &other) const {
  return getFirst() != other.getFirst() || getSecond() != other.getSecond();
}

llvm::raw_ostream &
DimensionAccessAdd::dump(llvm::raw_ostream &os,
                         const llvm::DenseMap<const IndexSet *, uint64_t>
                             &iterationSpacesIds) const {
  os << "(";
  getFirst().dump(os, iterationSpacesIds) << " + ";
  getSecond().dump(os, iterationSpacesIds) << ")";
  return os;
}

void DimensionAccessAdd::collectIterationSpaces(
    llvm::DenseSet<const IndexSet *> &iterationSpaces) const {
  getFirst().collectIterationSpaces(iterationSpaces);
  getSecond().collectIterationSpaces(iterationSpaces);
}

void DimensionAccessAdd::collectIterationSpaces(
    llvm::SmallVectorImpl<const IndexSet *> &iterationSpaces,
    llvm::DenseMap<const IndexSet *, llvm::DenseSet<uint64_t>>
        &dependentDimensions) const {
  getFirst().collectIterationSpaces(iterationSpaces, dependentDimensions);
  getSecond().collectIterationSpaces(iterationSpaces, dependentDimensions);
}

bool DimensionAccessAdd::isAffine() const {
  return getFirst().isAffine() && getSecond().isAffine();
}

mlir::AffineExpr DimensionAccessAdd::getAffineExpr() const {
  assert(isAffine());
  return getFirst().getAffineExpr() + getSecond().getAffineExpr();
}

mlir::AffineExpr DimensionAccessAdd::getAffineExpr(
    unsigned int numOfDimensions,
    DimensionAccess::FakeDimensionsMap &fakeDimensionsMap) const {
  mlir::AffineExpr firstExpr =
      getFirst().getAffineExpr(numOfDimensions, fakeDimensionsMap);

  mlir::AffineExpr secondExpr =
      getSecond().getAffineExpr(numOfDimensions, fakeDimensionsMap);

  return firstExpr + secondExpr;
}

IndexSet DimensionAccessAdd::map(
    const Point &point,
    llvm::DenseMap<const IndexSet *, Point> &currentIndexSetsPoint) const {
  const DimensionAccess &lhs = getFirst();
  const DimensionAccess &rhs = getSecond();

  IndexSet mappedLhs = lhs.map(point, currentIndexSetsPoint);
  IndexSet mappedRhs = rhs.map(point, currentIndexSetsPoint);

  IndexSet result;
  llvm::SmallVector<Point::data_type, 10> coordinates;

  for (Point mappedLhsPoint : mappedLhs) {
    for (Point mappedRhsPoint : mappedRhs) {
      assert(mappedLhsPoint.rank() == mappedRhsPoint.rank());
      coordinates.clear();

      for (size_t i = 0, e = mappedLhsPoint.rank(); i < e; ++i) {
        coordinates.push_back(mappedLhsPoint[i] + mappedRhsPoint[i]);
      }

      result += Point(coordinates);
    }
  }

  return result;
}

DimensionAccess &DimensionAccessAdd::getFirst() {
  assert(first && "First operand not set");
  return *first;
}

const DimensionAccess &DimensionAccessAdd::getFirst() const {
  assert(first && "First operand not set");
  return *first;
}

DimensionAccess &DimensionAccessAdd::getSecond() {
  assert(second && "Second operand not set");
  return *second;
}

const DimensionAccess &DimensionAccessAdd::getSecond() const {
  assert(second && "Second operand not set");
  return *second;
}
} // namespace marco::modeling
