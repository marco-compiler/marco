#ifndef MARCO_MODELING_INDEXSETRTREE_H
#define MARCO_MODELING_INDEXSETRTREE_H

#include "marco/Modeling/IndexSetImpl.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace marco::modeling::impl {
/// R-Tree IndexSet implementation.
class RTreeIndexSet : public IndexSet::Impl {
private:
  class PointIterator;
  class RangeIterator;

public:
  /// A node of the R-Tree.
  class Node {
  public:
    Node(Node *parent, const MultidimensionalRange &boundary);

    Node(const Node &other);

    ~Node();

    Node &operator=(Node &&other);

    bool isRoot() const;

    bool isLeaf() const;

    const MultidimensionalRange &getBoundary() const;

    /// Recalculate the MBR containing all the values / children.
    /// Return true if the boundary changed, false otherwise.
    bool recalcBoundary();

    /// Get the multidimensional space (area, volume, etc.) covered by
    /// the MBR.
    unsigned int getFlatSize() const;

    /// Get the amount of children or contained values.
    size_t fanOut() const;

    /// Add a child node.
    void add(std::unique_ptr<Node> child);

    /// Add a value.
    void add(MultidimensionalRange value);

  public:
    Node *parent;

  private:
    /// MBR containing all the children / values.
    MultidimensionalRange boundary;

    // The flat size of the MBR is stored internally so that it doesn't
    // need to be computed all the times (which has an O(d) cost, with d
    // equal to the rank).
    unsigned int flatSize;

  public:
    llvm::SmallVector<std::unique_ptr<Node>> children;
    llvm::SmallVector<MultidimensionalRange> values;
  };

  using const_point_iterator = IndexSet::const_point_iterator;
  using const_range_iterator = IndexSet::const_range_iterator;

  RTreeIndexSet();

  RTreeIndexSet(size_t minElements, size_t maxElements);

  RTreeIndexSet(llvm::ArrayRef<Point> points);

  RTreeIndexSet(llvm::ArrayRef<MultidimensionalRange> ranges);

  RTreeIndexSet(const RTreeIndexSet &other);

  RTreeIndexSet(RTreeIndexSet &&other);

  ~RTreeIndexSet() override;

  RTreeIndexSet &operator=(const RTreeIndexSet &other) = delete;

  RTreeIndexSet &operator=(RTreeIndexSet &&other) = delete;

  /// @name LLVM-style RTTI methods
  /// {

  static bool classof(const IndexSet::Impl *obj) {
    return obj->getKind() == RTree;
  }

  /// }

  std::unique_ptr<Impl> clone() const override;

  llvm::raw_ostream &dump(llvm::raw_ostream &os) const override;

  bool operator==(const Point &rhs) const override;

  bool operator==(const MultidimensionalRange &rhs) const override;

  bool operator==(const IndexSet::Impl &rhs) const override;

  bool operator==(const RTreeIndexSet &rhs) const;

  bool operator!=(const Point &rhs) const override;

  bool operator!=(const MultidimensionalRange &rhs) const override;

  bool operator!=(const IndexSet::Impl &rhs) const override;

  bool operator!=(const RTreeIndexSet &rhs) const;

  IndexSet::Impl &operator+=(const Point &rhs) override;

  IndexSet::Impl &operator+=(const MultidimensionalRange &rhs) override;

  IndexSet::Impl &operator+=(const IndexSet::Impl &rhs) override;

  IndexSet::Impl &operator+=(const RTreeIndexSet &rhs);

  IndexSet::Impl &operator-=(const Point &rhs) override;

  IndexSet::Impl &operator-=(const MultidimensionalRange &rhs) override;

  IndexSet::Impl &operator-=(const IndexSet::Impl &rhs) override;

  IndexSet::Impl &operator-=(const RTreeIndexSet &rhs);

  bool empty() const override;

  size_t rank() const override;

  size_t flatSize() const override;

  void clear() override;

  const_point_iterator begin() const override;

  const_point_iterator end() const override;

  const_range_iterator rangesBegin() const override;

  const_range_iterator rangesEnd() const override;

  bool contains(const Point &other) const override;

  bool contains(const MultidimensionalRange &other) const override;

  bool contains(const IndexSet::Impl &other) const override;

  bool contains(const RTreeIndexSet &other) const;

  bool overlaps(const MultidimensionalRange &other) const override;

  bool overlaps(const IndexSet::Impl &other) const override;

  bool overlaps(const RTreeIndexSet &other) const;

  IndexSet intersect(const MultidimensionalRange &other) const override;

  IndexSet intersect(const IndexSet::Impl &other) const override;

  IndexSet intersect(const RTreeIndexSet &other) const;

  IndexSet complement(const MultidimensionalRange &other) const override;

  std::unique_ptr<IndexSet::Impl>
  slice(const llvm::BitVector &filter) const override;

  std::unique_ptr<IndexSet::Impl> takeFirstDimensions(size_t n) const override;

  std::unique_ptr<IndexSet::Impl> takeLastDimensions(size_t n) const override;

  std::unique_ptr<IndexSet::Impl>
  takeDimensions(const llvm::SmallBitVector &dimensions) const override;

  std::unique_ptr<IndexSet::Impl> dropFirstDimensions(size_t n) const override;

  std::unique_ptr<IndexSet::Impl> dropLastDimensions(size_t n) const override;

  std::unique_ptr<IndexSet::Impl> prepend(const IndexSet &other) const override;

  std::unique_ptr<IndexSet::Impl> append(const IndexSet &other) const override;

  std::unique_ptr<IndexSet::Impl> getCanonicalRepresentation() const override;

  const Node *getRoot() const;

private:
  void setFromRanges(llvm::ArrayRef<MultidimensionalRange> ranges);

  Node *setRoot(std::unique_ptr<Node> node);

  /// Select a leaf node in which to place a new entry.
  Node *chooseLeaf(const MultidimensionalRange &entry) const;

  std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>> splitNode(Node &node);

  /// Check and adjust the tree structure so that each node has between a
  /// minimum and maximum amount of children.
  /// The values contained in collapsed nodes are given back to the caller for
  /// reinsertion.
  void adjustTree(Node *node,
                  llvm::SmallVectorImpl<MultidimensionalRange> &toReinsert);

  /// Check if all the invariants are respected.
  bool isValid() const;

  bool containsGenericIndexSet(const IndexSet::Impl &other) const;

  bool overlapsGenericIndexSet(const IndexSet::Impl &other) const;

public:
  /// The minimum number of elements for each node (apart the root).
  const size_t minElements;

  /// The maximum number of elements for each node.
  const size_t maxElements;

private:
  std::unique_ptr<Node> root;
  bool initialized;
  size_t allowedRank;
};
} // namespace marco::modeling::impl

#endif // MARCO_MODELING_INDEXSETRTREE_H
