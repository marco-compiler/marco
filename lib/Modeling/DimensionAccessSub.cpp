#include "marco/Modeling/DimensionAccessSub.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  DimensionAccessSub::DimensionAccessSub(
      mlir::MLIRContext* context,
      std::unique_ptr<DimensionAccess> first,
      std::unique_ptr<DimensionAccess> second)
      : DimensionAccess(DimensionAccess::Kind::Sub, context),
        first(std::move(first)),
        second(std::move(second))
  {
  }

  DimensionAccessSub::DimensionAccessSub(const DimensionAccessSub& other)
      : DimensionAccess(other),
        first(other.getFirst().clone()),
        second(other.getSecond().clone())
  {
  }

  DimensionAccessSub::DimensionAccessSub(DimensionAccessSub&& other) = default;

  DimensionAccessSub::~DimensionAccessSub() = default;

  DimensionAccessSub& DimensionAccessSub::operator=(
      const DimensionAccessSub& other)
  {
    DimensionAccessSub result(other);
    swap(*this, result);
    return *this;
  }

  DimensionAccessSub& DimensionAccessSub::operator=(
      DimensionAccessSub&& other) = default;

  void swap(DimensionAccessSub& first, DimensionAccessSub& second)
  {
    using std::swap;

    swap(static_cast<DimensionAccess&>(first),
         static_cast<DimensionAccess&>(second));

    swap(first.first, second.first);
    swap(first.second, second.second);
  }

  std::unique_ptr<DimensionAccess> DimensionAccessSub::clone() const
  {
    return std::make_unique<DimensionAccessSub>(*this);
  }

  bool DimensionAccessSub::operator==(const DimensionAccess& other) const
  {
    if (auto otherCasted = other.dyn_cast<DimensionAccessSub>()) {
      return *this == *otherCasted;
    }

    return false;
  }

  bool DimensionAccessSub::operator==(const DimensionAccessSub& other) const
  {
    return getFirst() == other.getFirst() && getSecond() == other.getSecond();
  }

  bool DimensionAccessSub::operator!=(const DimensionAccess& other) const
  {
    if (auto otherCasted = other.dyn_cast<DimensionAccessSub>()) {
      return *this != *otherCasted;
    }

    return true;
  }

  bool DimensionAccessSub::operator!=(const DimensionAccessSub& other) const
  {
    return getFirst() != other.getFirst() || getSecond() != other.getSecond();
  }

  llvm::raw_ostream& DimensionAccessSub::dump(llvm::raw_ostream& os) const
  {
    return os << "(" << getFirst() << " - " << getSecond() << ")";
  }

  bool DimensionAccessSub::isAffine() const
  {
    return getFirst().isAffine() && getSecond().isAffine();
  }

  mlir::AffineExpr DimensionAccessSub::getAffineExpr() const
  {
    assert(isAffine());
    return getFirst().getAffineExpr() - getSecond().getAffineExpr();
  }

  mlir::AffineExpr DimensionAccessSub::getAffineExpr(
      unsigned int numOfDimensions,
      DimensionAccess::FakeDimensionsMap& fakeDimensionsMap) const
  {
    mlir::AffineExpr firstExpr =
        getFirst().getAffineExpr(numOfDimensions, fakeDimensionsMap);

    mlir::AffineExpr secondExpr =
        getSecond().getAffineExpr(numOfDimensions, fakeDimensionsMap);

    return firstExpr - secondExpr;
  }

  IndexSet DimensionAccessSub::map(const Point& point) const
  {
    const DimensionAccess& lhs = getFirst();
    const DimensionAccess& rhs = getSecond();

    IndexSet mappedLhs = lhs.map(point);
    IndexSet mappedRhs = rhs.map(point);

    IndexSet result;
    llvm::SmallVector<Point::data_type, 10> coordinates;

    for (Point mappedLhsPoint : mappedLhs) {
      for (Point mappedRhsPoint : mappedRhs) {
        assert(mappedLhsPoint.rank() == mappedRhsPoint.rank());
        coordinates.clear();

        for (size_t i = 0, e = mappedLhsPoint.rank(); i < e; ++i) {
          coordinates.push_back(mappedLhsPoint[i] - mappedRhsPoint[i]);
        }

        result += Point(coordinates);
      }
    }

    return result;
  }

  DimensionAccess& DimensionAccessSub::getFirst()
  {
    assert(first && "First operand not set");
    return *first;
  }

  const DimensionAccess& DimensionAccessSub::getFirst() const
  {
    assert(first && "First operand not set");
    return *first;
  }

  DimensionAccess& DimensionAccessSub::getSecond()
  {
    assert(second && "Second operand not set");
    return *second;
  }

  const DimensionAccess& DimensionAccessSub::getSecond() const
  {
    assert(second && "Second operand not set");
    return *second;
  }
}
