#ifndef MARCO_MODELING_INDEXSETIMPL_H
#define MARCO_MODELING_INDEXSETIMPL_H

#include "marco/Modeling/IndexSet.h"
#include "llvm/Support/Casting.h"
#include <memory>

using namespace ::marco;
using namespace ::marco::modeling;

namespace marco::modeling {
namespace impl {
enum IndexSetKind { List, RTree };
}

class IndexSet::Impl {
public:
  using const_point_iterator = IndexSet::const_point_iterator;
  using const_range_iterator = IndexSet::const_range_iterator;

  Impl(impl::IndexSetKind kind);

  Impl(const Impl &other);

  Impl(Impl &&other);

  virtual ~Impl();

  Impl &operator=(const Impl &other);

  Impl &operator=(Impl &&other);

  friend std::ostream &operator<<(std::ostream &os, const IndexSet::Impl &obj);

  virtual std::unique_ptr<Impl> clone() const = 0;

  /// @name LLVM-style RTTI methods
  /// {

  impl::IndexSetKind getKind() const { return kind; }

  template <typename T>
  bool isa() const {
    return llvm::isa<T>(this);
  }

  template <typename T>
  T *dyn_cast() {
    return llvm::dyn_cast<T>(this);
  }

  template <typename T>
  const T *dyn_cast() const {
    return llvm::dyn_cast<T>(this);
  }

  /// }

  virtual llvm::raw_ostream &dump(llvm::raw_ostream &os) const = 0;

  virtual bool operator==(const Point &rhs) const = 0;

  virtual bool operator==(const MultidimensionalRange &rhs) const = 0;

  virtual bool operator==(const IndexSet::Impl &rhs) const = 0;

  virtual bool operator!=(const Point &rhs) const = 0;

  virtual bool operator!=(const MultidimensionalRange &rhs) const = 0;

  virtual bool operator!=(const IndexSet::Impl &rhs) const = 0;

  virtual Impl &operator+=(const Point &rhs) = 0;

  virtual Impl &operator+=(const MultidimensionalRange &rhs) = 0;

  virtual Impl &operator+=(const IndexSet::Impl &rhs) = 0;

  virtual Impl &operator-=(const Point &rhs) = 0;

  virtual Impl &operator-=(const MultidimensionalRange &rhs) = 0;

  virtual Impl &operator-=(const IndexSet::Impl &rhs) = 0;

  virtual bool empty() const = 0;

  virtual size_t rank() const = 0;

  virtual size_t flatSize() const = 0;

  virtual void clear() = 0;

  virtual const_point_iterator begin() const = 0;

  virtual const_point_iterator end() const = 0;

  virtual const_range_iterator rangesBegin() const = 0;

  virtual const_range_iterator rangesEnd() const = 0;

  virtual int compare(const IndexSet::Impl &other) const = 0;

  int compareGenericIndexSet(const IndexSet::Impl &other) const;

  virtual bool contains(const Point &other) const = 0;

  virtual bool contains(const MultidimensionalRange &other) const = 0;

  virtual bool contains(const IndexSet::Impl &other) const = 0;

  virtual bool overlaps(const MultidimensionalRange &other) const = 0;

  virtual bool overlaps(const IndexSet::Impl &other) const = 0;

  virtual IndexSet intersect(const MultidimensionalRange &other) const = 0;

  virtual IndexSet intersect(const IndexSet::Impl &other) const = 0;

  virtual IndexSet complement(const MultidimensionalRange &other) const = 0;

  virtual std::unique_ptr<Impl> slice(const llvm::BitVector &filter) const = 0;

  virtual std::unique_ptr<Impl> takeFirstDimensions(size_t n) const = 0;

  virtual std::unique_ptr<Impl> takeLastDimensions(size_t n) const = 0;

  virtual std::unique_ptr<Impl>
  takeDimensions(const llvm::SmallBitVector &dimensions) const = 0;

  virtual std::unique_ptr<Impl> dropFirstDimensions(size_t n) const = 0;

  virtual std::unique_ptr<Impl> dropLastDimensions(size_t n) const = 0;

  virtual std::unique_ptr<IndexSet::Impl>
  prepend(const IndexSet &other) const = 0;

  virtual std::unique_ptr<IndexSet::Impl>
  append(const IndexSet &other) const = 0;

  virtual std::unique_ptr<Impl> getCanonicalRepresentation() const = 0;

private:
  const impl::IndexSetKind kind;
};

class IndexSet::PointIterator::Impl {
public:
  using iterator_category = std::input_iterator_tag;
  using value_type = Point;
  using difference_type = std::ptrdiff_t;
  using pointer = const Point *;
  using reference = const Point &;

  Impl(impl::IndexSetKind kind);

  Impl(const Impl &other);

  Impl(Impl &&other);

  virtual ~Impl();

  Impl &operator=(const Impl &other);

  Impl &operator=(Impl &&other);

  friend void swap(Impl &first, Impl &second);

  virtual std::unique_ptr<IndexSet::PointIterator::Impl> clone() const = 0;

  /// @name LLVM-style RTTI methods
  /// {

  impl::IndexSetKind getKind() const { return kind; }

  template <typename T>
  bool isa() const {
    return llvm::isa<T>(this);
  }

  template <typename T>
  T *dyn_cast() {
    return llvm::dyn_cast<T>(this);
  }

  template <typename T>
  const T *dyn_cast() const {
    return llvm::dyn_cast<T>(this);
  }

  /// }

  virtual bool operator==(const Impl &other) const = 0;

  virtual bool operator!=(const Impl &other) const = 0;

  virtual Impl &operator++() = 0;

  virtual IndexSet::PointIterator operator++(int) = 0;

  virtual value_type operator*() const = 0;

private:
  impl::IndexSetKind kind;
};

class IndexSet::RangeIterator::Impl {
public:
  using iterator_category = std::input_iterator_tag;
  using value_type = MultidimensionalRange;
  using difference_type = std::ptrdiff_t;
  using pointer = const MultidimensionalRange *;
  using reference = const MultidimensionalRange &;

  Impl(impl::IndexSetKind kind);

  Impl(const Impl &other);

  Impl(Impl &&other);

  virtual ~Impl();

  Impl &operator=(const Impl &other);

  friend void swap(Impl &first, Impl &second);

  virtual std::unique_ptr<IndexSet::RangeIterator::Impl> clone() const = 0;

  /// @name LLVM-style RTTI methods
  /// {

  impl::IndexSetKind getKind() const { return kind; }

  template <typename T>
  bool isa() const {
    return llvm::isa<T>(this);
  }

  template <typename T>
  T *dyn_cast() {
    return llvm::dyn_cast<T>(this);
  }

  template <typename T>
  const T *dyn_cast() const {
    return llvm::dyn_cast<T>(this);
  }

  /// }

  virtual bool operator==(const Impl &other) const = 0;

  virtual bool operator!=(const Impl &other) const = 0;

  virtual Impl &operator++() = 0;

  virtual IndexSet::RangeIterator operator++(int) = 0;

  virtual reference operator*() const = 0;

private:
  impl::IndexSetKind kind;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_INDEXSETIMPL_H
