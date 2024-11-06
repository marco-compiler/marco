#ifndef MARCO_MODELING_MULTIDIMENSIONALRANGE_H
#define MARCO_MODELING_MULTIDIMENSIONALRANGE_H

#include "marco/Modeling/Range.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
class raw_ostream;
}

namespace marco::modeling {
/// n-D range. Each dimension is half-open as the 1-D range.
class MultidimensionalRange {
private:
  class Iterator {
  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = Point;
    using difference_type = std::ptrdiff_t;
    using pointer = Point *;
    using reference = Point &;

    Iterator(llvm::ArrayRef<Range> ranges,
             llvm::function_ref<Range::const_iterator(const Range &)> initFn);

    bool operator==(const Iterator &it) const;
    bool operator!=(const Iterator &it) const;

    Iterator &operator++();
    Iterator operator++(int);

    value_type operator*() const;

  private:
    void fetchNext();

  private:
    llvm::SmallVector<Range::const_iterator, 3> beginIterators;
    llvm::SmallVector<Range::const_iterator, 3> currentIterators;
    llvm::SmallVector<Range::const_iterator, 3> endIterators;
    llvm::SmallVector<Point::data_type, 3> indices;
  };

public:
  using const_iterator = Iterator;

  MultidimensionalRange(llvm::ArrayRef<Range> ranges);

  MultidimensionalRange(Point point);

  friend llvm::hash_code hash_value(const MultidimensionalRange &value);

  bool operator==(const Point &other) const;

  bool operator==(const MultidimensionalRange &other) const;

  bool operator!=(const Point &other) const;

  bool operator!=(const MultidimensionalRange &other) const;

  bool operator<(const MultidimensionalRange &other) const;

  bool operator>(const MultidimensionalRange &other) const;

  Range &operator[](size_t index);

  const Range &operator[](size_t index) const;

  unsigned int rank() const;

  void getSizes(llvm::SmallVectorImpl<size_t> &sizes) const;

  unsigned int flatSize() const;

  bool contains(const Point &other) const;

  bool contains(const MultidimensionalRange &other) const;

  bool overlaps(const MultidimensionalRange &other) const;

  MultidimensionalRange intersect(const MultidimensionalRange &other) const;

  /// Check if two multidimensional ranges can be merged.
  ///
  /// @return a pair whose first element is whether the merge is possible
  /// and the second one is the dimension to be merged
  std::pair<bool, size_t> canBeMerged(const MultidimensionalRange &other) const;

  MultidimensionalRange merge(const MultidimensionalRange &other,
                              size_t dimension) const;

  std::vector<MultidimensionalRange>
  subtract(const MultidimensionalRange &other) const;

  const_iterator begin() const;

  const_iterator end() const;

  MultidimensionalRange slice(size_t dimensions) const;

  MultidimensionalRange slice(const llvm::BitVector &filter) const;

  MultidimensionalRange takeFirstDimensions(size_t n) const;

  MultidimensionalRange takeLastDimensions(size_t n) const;

  MultidimensionalRange
  takeDimensions(const llvm::SmallBitVector &dimensions) const;

  MultidimensionalRange dropFirstDimensions(size_t n) const;

  MultidimensionalRange dropLastDimensions(size_t n) const;

  MultidimensionalRange prepend(const MultidimensionalRange &other) const;

  MultidimensionalRange append(const MultidimensionalRange &other) const;

private:
  llvm::SmallVector<Range> ranges;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const MultidimensionalRange &obj);
} // namespace marco::modeling

#endif // MARCO_MODELING_MULTIDIMENSIONALRANGE_H
