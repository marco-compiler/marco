#ifndef MARCO_MODELING_INDEXSET_H
#define MARCO_MODELING_INDEXSET_H

#include "marco/Modeling/MultidimensionalRange.h"
#include "marco/Modeling/RTree.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"

namespace llvm {
class raw_ostream;
}

namespace marco::modeling {
/// R-Tree information specialization for the MultidimensionalRange class.
template <>
struct RTreeInfo<MultidimensionalRange> {
  static const MultidimensionalRange &
  getShape(const MultidimensionalRange &val);

  static bool isEqual(const MultidimensionalRange &first,
                      const MultidimensionalRange &second);

  static void dump(llvm::raw_ostream &os, const MultidimensionalRange &val);
};
} // namespace marco::modeling

namespace marco::modeling {
/// R-Tree IndexSet implementation.
class IndexSet
    : public r_tree::impl::RTreeCRTP<IndexSet, MultidimensionalRange> {
  class RangeIterator {
    using base_iterator = RTreeCRTP::const_object_iterator;
    base_iterator baseIterator;

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = MultidimensionalRange;
    using difference_type = std::ptrdiff_t;
    using pointer = const MultidimensionalRange *;
    using reference = const MultidimensionalRange &;

    RangeIterator(base_iterator baseIterator);

    /// @name Constructors.
    /// {

    static IndexSet::RangeIterator begin(const IndexSet &obj);

    static IndexSet::RangeIterator end(const IndexSet &obj);

    /// }

    bool operator==(const IndexSet::RangeIterator &other) const;

    bool operator!=(const IndexSet::RangeIterator &other) const;

    RangeIterator &operator++();

    IndexSet::RangeIterator operator++(int);

    reference operator*() const;
  };

  class PointIterator {
    IndexSet::RangeIterator currentRangeIt;
    IndexSet::RangeIterator endRangeIt;
    std::optional<MultidimensionalRange::const_iterator> currentPointIt;
    std::optional<MultidimensionalRange::const_iterator> endPointIt;

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = Point;
    using difference_type = std::ptrdiff_t;
    using pointer = const Point *;
    using reference = const Point &;

    /// @name Constructors.
    /// {

    static IndexSet::PointIterator begin(const IndexSet &indexSet);

    static IndexSet::PointIterator end(const IndexSet &indexSet);

    /// }

    bool operator==(const IndexSet::PointIterator &other) const;

    bool operator!=(const IndexSet::PointIterator &other) const;

    IndexSet::PointIterator &operator++();

    IndexSet::PointIterator operator++(int);

    value_type operator*() const;

  private:
    PointIterator(
        IndexSet::RangeIterator currentRangeIt,
        IndexSet::RangeIterator endRangeIt,
        std::optional<MultidimensionalRange::const_iterator> currentPointIt,
        std::optional<MultidimensionalRange::const_iterator> endPointIt);

    bool shouldProceed() const;

    void fetchNext();

    void advance();
  };

public:
  using const_range_iterator = RangeIterator;
  using const_point_iterator = PointIterator;

  using RTreeCRTP::RTreeCRTP;

  IndexSet(llvm::ArrayRef<Point> points);

  IndexSet(llvm::ArrayRef<MultidimensionalRange> ranges);

  using RTreeCRTP::operator==;

  bool operator==(const Point &other) const;

  bool operator==(const MultidimensionalRange &other) const;

  friend llvm::hash_code hash_value(const IndexSet &value);

  using RTreeCRTP::operator!=;

  bool operator!=(const Point &other) const;

  bool operator!=(const MultidimensionalRange &other) const;

  int compare(const IndexSet &other) const;

  void setFromRanges(llvm::ArrayRef<MultidimensionalRange> ranges);

  IndexSet &operator+=(Point other);

private:
  void insert(MultidimensionalRange other) override;

public:
  IndexSet &operator+=(MultidimensionalRange other);

  IndexSet &operator+=(const IndexSet &other);

  IndexSet operator+(Point other) const;

  IndexSet operator+(MultidimensionalRange other) const;

  IndexSet operator+(const IndexSet &other) const;

  IndexSet &operator-=(const Point &other);

private:
  void remove(const MultidimensionalRange &other) override;

public:
  IndexSet &operator-=(const MultidimensionalRange &other);

  IndexSet &operator-=(const IndexSet &other);

  IndexSet operator-(const Point &other) const;

  IndexSet operator-(const MultidimensionalRange &other) const;

  IndexSet operator-(const IndexSet &other) const;

  size_t flatSize() const;

  const_point_iterator begin() const;

  const_point_iterator end() const;

  const_range_iterator rangesBegin() const;

  const_range_iterator rangesEnd() const;

  using RTreeCRTP::contains;

  bool contains(const Point &other) const;

  bool contains(const MultidimensionalRange &other) const override;

  bool contains(const IndexSet &other) const;

  bool overlaps(const MultidimensionalRange &other) const;

  bool overlaps(const IndexSet &other) const;

  IndexSet intersect(const MultidimensionalRange &other) const;

  IndexSet intersect(const IndexSet &other) const;

  IndexSet complement(const MultidimensionalRange &other) const;

  IndexSet slice(const llvm::BitVector &filter) const;

  IndexSet takeFirstDimensions(size_t n) const;

  IndexSet takeLastDimensions(size_t n) const;

  IndexSet takeDimensions(const llvm::SmallBitVector &dimensions) const;

  IndexSet dropFirstDimensions(size_t n) const;

  IndexSet dropLastDimensions(size_t n) const;

  IndexSet prepend(const IndexSet &other) const;

  IndexSet append(const IndexSet &other) const;

  void
  getCompactRanges(llvm::SmallVectorImpl<MultidimensionalRange> &result) const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const IndexSet &indexSet);

protected:
  void postObjectInsertionHook(Node &node) override;

  void postObjectRemovalHook(Node &node) override;
};
} // namespace marco::modeling

namespace llvm {
template <>
struct DenseMapInfo<marco::modeling::IndexSet> {
  using Key = marco::modeling::IndexSet;

  static Key getEmptyKey();

  static Key getTombstoneKey();

  static unsigned getHashValue(const Key &val);

  static bool isEqual(const Key &lhs, const Key &rhs);
};
} // namespace llvm

#endif // MARCO_MODELING_INDEXSETRTREE_H
