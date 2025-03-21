#include "marco/Modeling/IndexSet.h"
#include "marco/Modeling/IndexSetList.h"
#include "marco/Modeling/IndexSetRTree.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco;
using namespace ::marco::modeling;

//===---------------------------------------------------------------------===//
// IndexSet: implementation interface
//===---------------------------------------------------------------------===//

namespace marco::modeling {
IndexSet::Impl::Impl(impl::IndexSetKind kind) : kind(kind) {}

IndexSet::Impl::Impl(const IndexSet::Impl &other) = default;

IndexSet::Impl::Impl(IndexSet::Impl &&other) = default;

IndexSet::Impl::~Impl() = default;

IndexSet::Impl &IndexSet::Impl::operator=(const IndexSet::Impl &other) {
  const_cast<impl::IndexSetKind &>(kind) = other.kind;
  return *this;
}

IndexSet::Impl &IndexSet::Impl::operator=(IndexSet::Impl &&other) {
  const_cast<impl::IndexSetKind &>(kind) = other.kind;
  return *this;
}

int IndexSet::Impl::compareGenericIndexSet(const IndexSet::Impl &other) const {
  auto firstCanonical = getCanonicalRepresentation();
  auto secondCanonical = other.getCanonicalRepresentation();

  auto firstIt = firstCanonical->rangesBegin();
  auto firstEndIt = firstCanonical->rangesEnd();

  auto secondIt = secondCanonical->rangesBegin();
  auto secondEndIt = secondCanonical->rangesEnd();

  while (firstIt != firstEndIt && secondIt != secondEndIt) {
    if (auto rangeCmp = (*firstIt).compare(*secondIt); rangeCmp != 0) {
      return rangeCmp;
    }

    ++firstIt;
    ++secondIt;
  }

  if (firstIt == firstEndIt && secondIt != secondEndIt) {
    // First set has fewer ranges.
    return -1;
  }

  if (firstIt != firstEndIt && secondIt == secondEndIt) {
    // Second set has fewer ranges.
    return 1;
  }

  assert(firstIt == firstEndIt && secondIt == secondEndIt);
  return 0;
}

std::ostream &operator<<(std::ostream &os, const IndexSet::Impl &obj) {
  if (auto *objCasted = obj.dyn_cast<impl::ListIndexSet>()) {
    return os << *objCasted;
  }

  if (auto *objCasted = obj.dyn_cast<impl::RTreeIndexSet>()) {
    return os << *objCasted;
  }

  return os;
}

llvm::hash_code hash_value(const IndexSet::Impl &value) {
  if (auto *objCasted = value.dyn_cast<impl::ListIndexSet>()) {
    return hash_value(*objCasted);
  }

  if (auto *objCasted = value.dyn_cast<impl::RTreeIndexSet>()) {
    return hash_value(*objCasted);
  }

  llvm_unreachable("Unknown IndexSet type");
  return {0};
}
} // namespace marco::modeling

//===---------------------------------------------------------------------===//
// IndexSet
//===---------------------------------------------------------------------===//

namespace marco::modeling {
IndexSet::IndexSet(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {
  assert(this->impl != nullptr);
}

IndexSet::IndexSet() : impl(std::make_unique<impl::RTreeIndexSet>()) {}

IndexSet::IndexSet(llvm::ArrayRef<Point> points)
    : impl(std::make_unique<impl::RTreeIndexSet>(points)) {}

IndexSet::IndexSet(llvm::ArrayRef<MultidimensionalRange> ranges)
    : impl(std::make_unique<impl::RTreeIndexSet>(ranges)) {}

IndexSet::IndexSet(const IndexSet &other) : impl(other.impl->clone()) {}

IndexSet::IndexSet(IndexSet &&other) = default;

IndexSet::~IndexSet() = default;

IndexSet &IndexSet::operator=(const IndexSet &other) {
  IndexSet result(other);
  swap(*this, result);
  return *this;
}

IndexSet &IndexSet::operator=(IndexSet &&other) = default;

void swap(IndexSet &first, IndexSet &second) {
  using std::swap;
  swap(first.impl, second.impl);
}

llvm::hash_code hash_value(const IndexSet &value) {
  assert(value.impl != nullptr);
  return hash_value(*value.impl);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const IndexSet &obj) {
  assert(obj.impl != nullptr);
  return obj.impl->dump(os);
}

llvm::raw_ostream &IndexSet::dump(llvm::raw_ostream &os) const {
  assert(impl != nullptr);
  return impl->dump(os);
}

bool IndexSet::operator==(const Point &rhs) const {
  assert(impl != nullptr);
  return *impl == rhs;
}

bool IndexSet::operator==(const MultidimensionalRange &rhs) const {
  assert(impl != nullptr);
  return *impl == rhs;
}

bool IndexSet::operator==(const IndexSet &rhs) const {
  assert(impl != nullptr);
  assert(rhs.impl != nullptr);
  return *impl == *rhs.impl;
}

bool IndexSet::operator!=(const Point &rhs) const {
  assert(impl != nullptr);
  return *impl == rhs;
}

bool IndexSet::operator!=(const MultidimensionalRange &rhs) const {
  assert(impl != nullptr);
  return *impl == rhs;
}

bool IndexSet::operator!=(const IndexSet &rhs) const {
  assert(impl != nullptr);
  assert(rhs.impl != nullptr);
  return *impl != *rhs.impl;
}

IndexSet &IndexSet::operator+=(const Point &rhs) {
  assert(impl != nullptr);
  *impl += rhs;
  return *this;
}

IndexSet &IndexSet::operator+=(const MultidimensionalRange &rhs) {
  assert(impl != nullptr);
  *impl += rhs;
  return *this;
}

IndexSet &IndexSet::operator+=(const IndexSet &rhs) {
  assert(impl != nullptr);
  assert(rhs.impl != nullptr);
  *impl += *rhs.impl;
  return *this;
}

IndexSet IndexSet::operator+(const Point &rhs) const {
  IndexSet result(*this);
  result += rhs;
  return result;
}

IndexSet IndexSet::operator+(const MultidimensionalRange &rhs) const {
  IndexSet result(*this);
  result += rhs;
  return result;
}

IndexSet IndexSet::operator+(const IndexSet &rhs) const {
  IndexSet result(*this);
  result += rhs;
  return result;
}

IndexSet &IndexSet::operator-=(const Point &rhs) {
  assert(impl != nullptr);
  *impl -= rhs;
  return *this;
}

IndexSet &IndexSet::operator-=(const MultidimensionalRange &rhs) {
  assert(impl != nullptr);
  *impl -= rhs;
  return *this;
}

IndexSet &IndexSet::operator-=(const IndexSet &rhs) {
  assert(impl != nullptr);
  assert(rhs.impl != nullptr);
  *impl -= *rhs.impl;
  return *this;
}

IndexSet IndexSet::operator-(const Point &rhs) const {
  IndexSet result(*this);
  result -= rhs;
  return result;
}

IndexSet IndexSet::operator-(const MultidimensionalRange &rhs) const {
  IndexSet result(*this);
  result -= rhs;
  return result;
}

IndexSet IndexSet::operator-(const IndexSet &rhs) const {
  IndexSet result(*this);
  result -= rhs;
  return result;
}

bool IndexSet::empty() const {
  assert(impl != nullptr);
  return impl->empty();
}

size_t IndexSet::rank() const {
  assert(impl != nullptr);
  return impl->rank();
}

size_t IndexSet::flatSize() const {
  assert(impl != nullptr);
  return impl->flatSize();
}

void IndexSet::clear() {
  assert(impl != nullptr);
  impl->clear();
}

IndexSet::const_point_iterator IndexSet::begin() const {
  assert(impl != nullptr);
  return impl->begin();
}

IndexSet::const_point_iterator IndexSet::end() const {
  assert(impl != nullptr);
  return impl->end();
}

IndexSet::const_range_iterator IndexSet::rangesBegin() const {
  assert(impl != nullptr);
  return impl->rangesBegin();
}

IndexSet::const_range_iterator IndexSet::rangesEnd() const {
  assert(impl != nullptr);
  return impl->rangesEnd();
}

int IndexSet::compare(const IndexSet &other) const {
  assert(impl != nullptr);
  return impl->compare(*other.impl);
}

bool IndexSet::contains(const Point &other) const {
  assert(impl != nullptr);
  return impl->contains(other);
}

bool IndexSet::contains(const MultidimensionalRange &other) const {
  assert(impl != nullptr);
  return impl->contains(other);
}

bool IndexSet::contains(const IndexSet &other) const {
  assert(impl != nullptr);
  assert(other.impl != nullptr);
  return impl->contains(*other.impl);
}

bool IndexSet::overlaps(const MultidimensionalRange &other) const {
  assert(impl != nullptr);
  return impl->overlaps(other);
}

bool IndexSet::overlaps(const IndexSet &other) const {
  assert(impl != nullptr);
  assert(other.impl != nullptr);
  return impl->overlaps(*other.impl);
}

IndexSet IndexSet::intersect(const MultidimensionalRange &other) const {
  assert(impl != nullptr);
  return impl->intersect(other);
}

IndexSet IndexSet::intersect(const IndexSet &other) const {
  assert(impl != nullptr);
  assert(other.impl != nullptr);
  return impl->intersect(*other.impl);
}

IndexSet IndexSet::complement(const MultidimensionalRange &other) const {
  assert(impl != nullptr);
  return impl->complement(other);
}

IndexSet IndexSet::slice(const llvm::BitVector &filter) const {
  assert(impl != nullptr);
  return {impl->slice(filter)};
}

IndexSet IndexSet::takeFirstDimensions(size_t n) const {
  assert(impl != nullptr);
  return {impl->takeFirstDimensions(n)};
}

IndexSet IndexSet::takeLastDimensions(size_t n) const {
  assert(impl != nullptr);
  return {impl->takeLastDimensions(n)};
}

IndexSet
IndexSet::takeDimensions(const llvm::SmallBitVector &dimensions) const {
  assert(impl != nullptr);
  return {impl->takeDimensions(dimensions)};
}

IndexSet IndexSet::dropFirstDimensions(size_t n) const {
  assert(impl != nullptr);
  return {impl->dropFirstDimensions(n)};
}

IndexSet IndexSet::dropLastDimensions(size_t n) const {
  assert(impl != nullptr);
  return {impl->dropLastDimensions(n)};
}

IndexSet IndexSet::prepend(const IndexSet &other) const {
  assert(impl != nullptr);
  return {impl->prepend(other)};
}

IndexSet IndexSet::append(const IndexSet &other) const {
  assert(impl != nullptr);
  return {impl->append(other)};
}

IndexSet IndexSet::getCanonicalRepresentation() const {
  assert(impl != nullptr);
  return {impl->getCanonicalRepresentation()};
}
} // namespace marco::modeling

//===---------------------------------------------------------------------===//
// Point iterator: implementation interface
//===---------------------------------------------------------------------===//

namespace marco::modeling {
IndexSet::PointIterator::Impl::Impl(impl::IndexSetKind kind) : kind(kind) {}

IndexSet::PointIterator::Impl::Impl(
    const IndexSet::PointIterator::Impl &other) = default;

IndexSet::PointIterator::Impl::Impl(IndexSet::PointIterator::Impl &&other) =
    default;

IndexSet::PointIterator::Impl::~Impl() = default;
} // namespace marco::modeling

//===---------------------------------------------------------------------===//
// Point iterator
//===---------------------------------------------------------------------===//

namespace marco::modeling {
IndexSet::PointIterator::PointIterator(std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

IndexSet::PointIterator::PointIterator(const IndexSet::PointIterator &other)
    : impl(other.impl == nullptr ? nullptr : other.impl->clone()) {}

IndexSet::PointIterator::PointIterator(IndexSet::PointIterator &&other) =
    default;

IndexSet::PointIterator::~PointIterator() = default;

IndexSet::PointIterator &
IndexSet::PointIterator::operator=(const IndexSet::PointIterator &other) {
  IndexSet::PointIterator result(other);
  swap(*this, result);
  return *this;
}

void swap(IndexSet::PointIterator &first, IndexSet::PointIterator &second) {
  using std::swap;
  swap(first.impl, second.impl);
}

bool IndexSet::PointIterator::operator==(
    const IndexSet::PointIterator &it) const {
  if (impl == it.impl) {
    return true;
  }

  if (!impl || !it.impl) {
    return false;
  }

  return *impl == *it.impl;
}

bool IndexSet::PointIterator::operator!=(
    const IndexSet::PointIterator &it) const {
  return !(*this == it);
}

IndexSet::PointIterator &IndexSet::PointIterator::operator++() {
  ++(*impl);
  return *this;
}

IndexSet::PointIterator IndexSet::PointIterator::operator++(int) {
  auto tmp = *this;
  ++(*impl);
  return tmp;
}

IndexSet::PointIterator::value_type IndexSet::PointIterator::operator*() const {
  return **impl;
}
} // namespace marco::modeling

//===---------------------------------------------------------------------===//
// Range iterator: implementation interface
//===---------------------------------------------------------------------===//

namespace marco::modeling {
IndexSet::RangeIterator::Impl::Impl(impl::IndexSetKind kind) : kind(kind) {}

IndexSet::RangeIterator::Impl::Impl(
    const IndexSet::RangeIterator::Impl &other) = default;

IndexSet::RangeIterator::Impl::Impl(IndexSet::RangeIterator::Impl &&other) =
    default;

IndexSet::RangeIterator::Impl::~Impl() = default;
} // namespace marco::modeling

//===---------------------------------------------------------------------===//
// Range Iterator
//===---------------------------------------------------------------------===//

namespace marco::modeling {
IndexSet::RangeIterator::RangeIterator(std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

IndexSet::RangeIterator::RangeIterator(const IndexSet::RangeIterator &other)
    : impl(other.impl == nullptr ? nullptr : other.impl->clone()) {}

IndexSet::RangeIterator::RangeIterator(IndexSet::RangeIterator &&other) =
    default;

IndexSet::RangeIterator::~RangeIterator() = default;

IndexSet::RangeIterator &
IndexSet::RangeIterator::operator=(const IndexSet::RangeIterator &other) {
  IndexSet::RangeIterator result(other);
  swap(*this, result);
  return *this;
}

void swap(IndexSet::RangeIterator &first, IndexSet::RangeIterator &second) {
  using std::swap;
  swap(first.impl, second.impl);
}

bool IndexSet::RangeIterator::operator==(
    const IndexSet::RangeIterator &it) const {
  if (impl == it.impl) {
    return true;
  }

  if (!impl || !it.impl) {
    return false;
  }

  return *impl == *it.impl;
}

bool IndexSet::RangeIterator::operator!=(
    const IndexSet::RangeIterator &it) const {
  return !(*this == it);
}

IndexSet::RangeIterator &IndexSet::RangeIterator::operator++() {
  ++(*impl);
  return *this;
}

IndexSet::RangeIterator IndexSet::RangeIterator::operator++(int) {
  IndexSet::RangeIterator temp = *this;
  ++(*impl);
  return temp;
}

IndexSet::RangeIterator::reference IndexSet::RangeIterator::operator*() const {
  return **impl;
}
} // namespace marco::modeling
