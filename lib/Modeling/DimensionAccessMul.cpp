#include "marco/Modeling/DimensionAccessMul.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  DimensionAccessMul::DimensionAccessMul(
      mlir::MLIRContext* context,
      std::unique_ptr<DimensionAccess> first,
      std::unique_ptr<DimensionAccess> second)
      : DimensionAccess(DimensionAccess::Kind::Mul, context),
        first(std::move(first)),
        second(std::move(second))
  {
  }

  DimensionAccessMul::DimensionAccessMul(const DimensionAccessMul& other)
      : DimensionAccess(other),
        first(other.getFirst().clone()),
        second(other.getSecond().clone())
  {
  }

  DimensionAccessMul::DimensionAccessMul(
      DimensionAccessMul&& other) noexcept = default;

  DimensionAccessMul::~DimensionAccessMul() = default;

  DimensionAccessMul& DimensionAccessMul::operator=(
      const DimensionAccessMul& other)
  {
    DimensionAccessMul result(other);
    swap(*this, result);
    return *this;
  }

  DimensionAccessMul& DimensionAccessMul::operator=(
      DimensionAccessMul&& other) noexcept = default;

  void swap(DimensionAccessMul& first, DimensionAccessMul& second)
  {
    using std::swap;

    swap(static_cast<DimensionAccess&>(first),
         static_cast<DimensionAccess&>(second));

    swap(first.first, second.first);
    swap(first.second, second.second);
  }

  std::unique_ptr<DimensionAccess> DimensionAccessMul::clone() const
  {
    return std::make_unique<DimensionAccessMul>(*this);
  }

  bool DimensionAccessMul::operator==(const DimensionAccess& other) const
  {
    if (auto otherCasted = other.dyn_cast<DimensionAccessMul>()) {
      return *this == *otherCasted;
    }

    return false;
  }

  bool DimensionAccessMul::operator==(const DimensionAccessMul& other) const
  {
    return getFirst() == other.getFirst() && getSecond() == other.getSecond();
  }

  bool DimensionAccessMul::operator!=(const DimensionAccess& other) const
  {
    if (auto otherCasted = other.dyn_cast<DimensionAccessMul>()) {
      return *this != *otherCasted;
    }

    return true;
  }

  bool DimensionAccessMul::operator!=(const DimensionAccessMul& other) const
  {
    return getFirst() != other.getFirst() || getSecond() != other.getSecond();
  }

  llvm::raw_ostream& DimensionAccessMul::dump(
      llvm::raw_ostream& os,
      const llvm::DenseMap<
          const IndexSet*, uint64_t>& iterationSpacesIds) const
  {
    os << "(";
    getFirst().dump(os, iterationSpacesIds) << " * ";
    getSecond().dump(os, iterationSpacesIds) << ")";
    return os;
  }

  void DimensionAccessMul::collectIterationSpaces(
      llvm::DenseSet<const IndexSet*>& iterationSpaces) const
  {
    getFirst().collectIterationSpaces(iterationSpaces);
    getSecond().collectIterationSpaces(iterationSpaces);
  }

  void DimensionAccessMul::collectIterationSpaces(
      llvm::SmallVectorImpl<const IndexSet*>& iterationSpaces,
      llvm::DenseMap<
          const IndexSet*,
          llvm::DenseSet<uint64_t>>& dependentDimensions) const
  {
    getFirst().collectIterationSpaces(iterationSpaces, dependentDimensions);
    getSecond().collectIterationSpaces(iterationSpaces, dependentDimensions);
  }

  bool DimensionAccessMul::isAffine() const
  {
    return getFirst().isAffine() && getSecond().isAffine();
  }

  mlir::AffineExpr DimensionAccessMul::getAffineExpr() const
  {
    assert(isAffine());
    return getFirst().getAffineExpr() * getSecond().getAffineExpr();
  }

  mlir::AffineExpr DimensionAccessMul::getAffineExpr(
      unsigned int numOfDimensions,
      DimensionAccess::FakeDimensionsMap& fakeDimensionsMap) const
  {
    mlir::AffineExpr firstExpr =
        getFirst().getAffineExpr(numOfDimensions, fakeDimensionsMap);

    mlir::AffineExpr secondExpr =
        getSecond().getAffineExpr(numOfDimensions, fakeDimensionsMap);

    return firstExpr * secondExpr;
  }

  IndexSet DimensionAccessMul::map(
      const Point& point,
      llvm::DenseMap<const IndexSet*, Point>& currentIndexSetsPoint) const
  {
    const DimensionAccess& lhs = getFirst();
    const DimensionAccess& rhs = getSecond();

    IndexSet mappedLhs = lhs.map(point, currentIndexSetsPoint);
    IndexSet mappedRhs = rhs.map(point, currentIndexSetsPoint);

    IndexSet result;
    llvm::SmallVector<Point::data_type, 10> coordinates;

    for (Point mappedLhsPoint : mappedLhs) {
      for (Point mappedRhsPoint : mappedRhs) {
        assert(mappedLhsPoint.rank() == mappedRhsPoint.rank());
        coordinates.clear();

        for (size_t i = 0, e = mappedLhsPoint.rank(); i < e; ++i) {
          coordinates.push_back(mappedLhsPoint[i] * mappedRhsPoint[i]);
        }

        result += Point(coordinates);
      }
    }

    return result;
  }

  DimensionAccess& DimensionAccessMul::getFirst()
  {
    assert(first && "First operand not set");
    return *first;
  }

  const DimensionAccess& DimensionAccessMul::getFirst() const
  {
    assert(first && "First operand not set");
    return *first;
  }

  DimensionAccess& DimensionAccessMul::getSecond()
  {
    assert(second && "Second operand not set");
    return *second;
  }

  const DimensionAccess& DimensionAccessMul::getSecond() const
  {
    assert(second && "Second operand not set");
    return *second;
  }
}
