#include "marco/Modeling/IndexSetRTree.h"
#include "marco/Modeling/IndexSetList.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <queue>

using namespace ::marco::modeling;
using namespace ::marco::modeling::impl;

namespace {
void merge(llvm::SmallVector<MultidimensionalRange> &ranges) {
  if (ranges.empty()) {
    return;
  }

  using It = llvm::SmallVector<MultidimensionalRange>::iterator;

  auto findCandidates = [&](It begin, It end) -> std::tuple<It, It, size_t> {
    for (It it1 = begin; it1 != end; ++it1) {
      for (It it2 = std::next(it1); it2 != end; ++it2) {
        if (auto mergePossibility = it1->canBeMerged(*it2);
            mergePossibility.first) {
          return std::make_tuple(it1, it2, mergePossibility.second);
        }
      }
    }

    return std::make_tuple(end, end, 0);
  };

  auto candidates = findCandidates(ranges.begin(), ranges.end());

  while (std::get<0>(candidates) != ranges.end() &&
         std::get<1>(candidates) != ranges.end()) {
    auto &first = std::get<0>(candidates);
    auto &second = std::get<1>(candidates);
    size_t dimension = std::get<2>(candidates);

    *first = first->merge(*second, dimension);
    ranges.erase(second);
    candidates = findCandidates(ranges.begin(), ranges.end());
  }

  assert(!ranges.empty());
}

template <typename T>
const MultidimensionalRange &getShape(const T &obj);

template <>
const MultidimensionalRange &
getShape<MultidimensionalRange>(const MultidimensionalRange &range) {
  return range;
}

template <>
const MultidimensionalRange &getShape<std::unique_ptr<RTreeIndexSet::Node>>(
    const std::unique_ptr<RTreeIndexSet::Node> &node) {
  return node->getBoundary();
}

/// Get the minimum bounding (multidimensional) range.
MultidimensionalRange getMBR(const MultidimensionalRange &first,
                             const MultidimensionalRange &second) {
  assert(first.rank() == second.rank() &&
         "Can't compute the MBR between ranges on two different "
         "hyper-spaces");

  std::vector<Range> ranges;

  for (size_t i = 0; i < first.rank(); ++i) {
    ranges.emplace_back(std::min(first[i].getBegin(), second[i].getBegin()),
                        std::max(first[i].getEnd(), second[i].getEnd()));
  }

  return {ranges};
}

template <typename T>
MultidimensionalRange getMBR(llvm::ArrayRef<T> objects) {
  assert(!objects.empty());
  MultidimensionalRange result = getShape(objects[0]);

  for (size_t i = 1; i < objects.size(); ++i) {
    result = getMBR(result, getShape(objects[i]));
  }

  return result;
}
} // namespace

//===---------------------------------------------------------------------===//
// R-Tree node
//===---------------------------------------------------------------------===//

namespace marco::modeling::impl {
RTreeIndexSet::Node::Node(Node *parent, const MultidimensionalRange &boundary)
    : parent(parent), boundary(boundary), flatSize(boundary.flatSize()) {
  assert(this != parent && "A node can't be the parent of itself");
}

RTreeIndexSet::Node::Node(const Node &other)
    : parent(other.parent), boundary(other.boundary), flatSize(other.flatSize) {
  for (const auto &child : other.children) {
    auto &newChild = children.emplace_back(std::make_unique<Node>(*child));
    newChild->parent = this;
  }

  for (const auto &value : other.values) {
    values.push_back(value);
  }
}

RTreeIndexSet::Node::~Node() = default;

RTreeIndexSet::Node &RTreeIndexSet::Node::operator=(Node &&other) = default;

bool RTreeIndexSet::Node::isRoot() const { return parent == nullptr; }

bool RTreeIndexSet::Node::isLeaf() const {
  auto result = children.empty();
  assert(result || values.empty());
  return result;
}

const MultidimensionalRange &RTreeIndexSet::Node::getBoundary() const {
  return boundary;
}

void RTreeIndexSet::Node::recalcBoundary() {
  if (isLeaf()) {
    boundary = getMBR(llvm::ArrayRef(values));
  } else {
    boundary = getMBR(llvm::ArrayRef(children));
  }

  // Check the validity of the MBR
  assert(llvm::all_of(children, [&](const auto &child) {
    return boundary.contains(getShape(child));
  }));

  assert(llvm::all_of(values, [&](const auto &value) {
    return boundary.contains(getShape(value));
  }));

  // Update the covered area
  flatSize = boundary.flatSize();
}

unsigned int RTreeIndexSet::Node::getFlatSize() const { return flatSize; }

size_t RTreeIndexSet::Node::fanOut() const {
  if (isLeaf()) {
    return values.size();
  }

  return children.size();
}

void RTreeIndexSet::Node::add(std::unique_ptr<Node> child) {
  assert(values.empty());
  children.push_back(std::move(child));
}

void RTreeIndexSet::Node::add(MultidimensionalRange value) {
  assert(children.empty());
  values.push_back(std::move(value));
}
} // namespace marco::modeling::impl

//===---------------------------------------------------------------------===//
// R-Tree-IndexSet: point iterator
//===---------------------------------------------------------------------===//

namespace marco::modeling::impl {
class RTreeIndexSet::PointIterator : public IndexSet::PointIterator::Impl {
public:
  using iterator_category = IndexSet::PointIterator::Impl::iterator_category;

  using value_type = IndexSet::PointIterator::Impl::value_type;
  using difference_type = IndexSet::PointIterator::Impl::difference_type;
  using pointer = IndexSet::PointIterator::Impl::pointer;
  using reference = IndexSet::PointIterator::Impl::reference;

  /// @name LLVM-style RTTI methods
  /// {

  static bool classof(const IndexSet::PointIterator::Impl *obj) {
    return obj->getKind() == RTree;
  }

  /// }

  std::unique_ptr<IndexSet::PointIterator::Impl> clone() const override;

  /// @name Construction methods
  /// {

  static IndexSet::PointIterator begin(const RTreeIndexSet &indexSet);

  static IndexSet::PointIterator end(const RTreeIndexSet &indexSet);

  /// }

  bool operator==(const IndexSet::PointIterator::Impl &other) const override;

  bool operator!=(const IndexSet::PointIterator::Impl &other) const override;

  IndexSet::PointIterator::Impl &operator++() override;

  IndexSet::PointIterator operator++(int) override;

  value_type operator*() const override;

private:
  PointIterator(
      IndexSet::RangeIterator currentRangeIt,
      IndexSet::RangeIterator endRangeIt,
      std::optional<MultidimensionalRange::const_iterator> currentPointIt,
      std::optional<MultidimensionalRange::const_iterator> endPointIt);

  bool shouldProceed() const;

  void fetchNext();

  void advance();

private:
  IndexSet::RangeIterator currentRangeIt;
  IndexSet::RangeIterator endRangeIt;
  std::optional<MultidimensionalRange::const_iterator> currentPointIt;
  std::optional<MultidimensionalRange::const_iterator> endPointIt;
};

RTreeIndexSet::PointIterator::PointIterator(
    IndexSet::RangeIterator currentRangeIt, IndexSet::RangeIterator endRangeIt,
    std::optional<MultidimensionalRange::const_iterator> currentPointIt,
    std::optional<MultidimensionalRange::const_iterator> endPointIt)
    : IndexSet::PointIterator::Impl(RTree),
      currentRangeIt(std::move(currentRangeIt)),
      endRangeIt(std::move(endRangeIt)),
      currentPointIt(std::move(currentPointIt)),
      endPointIt(std::move(endPointIt)) {
  fetchNext();
}

std::unique_ptr<IndexSet::PointIterator::Impl>
RTreeIndexSet::PointIterator::clone() const {
  return std::make_unique<RTreeIndexSet::PointIterator>(*this);
}

IndexSet::PointIterator
RTreeIndexSet::PointIterator::begin(const RTreeIndexSet &indexSet) {
  if (indexSet.empty()) {
    return {nullptr};
  }

  auto currentRangeIt = indexSet.rangesBegin();
  auto endRangeIt = indexSet.rangesEnd();

  if (currentRangeIt == endRangeIt) {
    // There are no ranges. The current range iterator is already
    // past-the-end, and thus we must avoid dereferencing it.

    RTreeIndexSet::PointIterator it(currentRangeIt, endRangeIt, std::nullopt,
                                    std::nullopt);

    return {std::make_unique<RTreeIndexSet::PointIterator>(std::move(it))};
  }

  auto currentPointIt = (*currentRangeIt).begin();
  auto endPointIt = (*currentRangeIt).end();

  RTreeIndexSet::PointIterator it(currentRangeIt, endRangeIt, currentPointIt,
                                  endPointIt);

  return {std::make_unique<RTreeIndexSet::PointIterator>(std::move(it))};
}

IndexSet::PointIterator
RTreeIndexSet::PointIterator::end(const RTreeIndexSet &indexSet) {
  if (indexSet.empty()) {
    return {nullptr};
  }

  RTreeIndexSet::PointIterator it(indexSet.rangesEnd(), indexSet.rangesEnd(),
                                  std::nullopt, std::nullopt);

  return {std::make_unique<RTreeIndexSet::PointIterator>(std::move(it))};
}

bool RTreeIndexSet::PointIterator::operator==(
    const IndexSet::PointIterator::Impl &other) const {
  if (auto *it = other.dyn_cast<RTreeIndexSet::PointIterator>()) {
    return currentRangeIt == it->currentRangeIt &&
           currentPointIt == it->currentPointIt;
  }

  return false;
}

bool RTreeIndexSet::PointIterator::operator!=(
    const IndexSet::PointIterator::Impl &other) const {
  if (auto *it = other.dyn_cast<RTreeIndexSet::PointIterator>()) {
    return currentRangeIt != it->currentRangeIt ||
           currentPointIt != it->currentPointIt;
  }

  return true;
}

IndexSet::PointIterator::Impl &RTreeIndexSet::PointIterator::operator++() {
  advance();
  return *this;
}

IndexSet::PointIterator RTreeIndexSet::PointIterator::operator++(int) {
  IndexSet::PointIterator result(clone());
  ++(*this);
  return result;
}

RTreeIndexSet::PointIterator::value_type
RTreeIndexSet::PointIterator::operator*() const {
  return **currentPointIt;
}

bool RTreeIndexSet::PointIterator::shouldProceed() const {
  if (currentRangeIt == endRangeIt) {
    return false;
  }

  return currentPointIt == endPointIt;
}

void RTreeIndexSet::PointIterator::fetchNext() {
  while (shouldProceed()) {
    bool advanceToNextRange = currentPointIt == endPointIt;

    if (advanceToNextRange) {
      ++currentRangeIt;

      if (currentRangeIt == endRangeIt) {
        currentPointIt = std::nullopt;
        endPointIt = std::nullopt;
      } else {
        currentPointIt = (*currentRangeIt).begin();
        endPointIt = (*currentRangeIt).end();
      }
    } else {
      ++(*currentPointIt);
    }
  }
}

void RTreeIndexSet::PointIterator::advance() {
  ++(*currentPointIt);
  fetchNext();
}
} // namespace marco::modeling::impl

//===---------------------------------------------------------------------===//
// R-Tree-IndexSet: range iterator
//===---------------------------------------------------------------------===//

namespace marco::modeling::impl {
class RTreeIndexSet::RangeIterator : public IndexSet::RangeIterator::Impl {
public:
  using iterator_category = IndexSet::RangeIterator::Impl::iterator_category;

  using value_type = IndexSet::RangeIterator::Impl::value_type;
  using difference_type = IndexSet::RangeIterator::Impl::difference_type;
  using pointer = IndexSet::RangeIterator::Impl::pointer;
  using reference = IndexSet::RangeIterator::Impl::reference;

  RangeIterator(const Node *root);

  /// @name LLVM-style RTTI methods
  /// {

  static bool classof(const IndexSet::RangeIterator::Impl *obj) {
    return obj->getKind() == RTree;
  }

  /// }

  std::unique_ptr<IndexSet::RangeIterator::Impl> clone() const override;

  /// @name Construction methods
  /// {

  static IndexSet::RangeIterator begin(const RTreeIndexSet &indexSet);

  static IndexSet::RangeIterator end(const RTreeIndexSet &indexSet);

  /// }

  bool operator==(const IndexSet::RangeIterator::Impl &other) const override;

  bool operator!=(const IndexSet::RangeIterator::Impl &other) const override;

  IndexSet::RangeIterator::Impl &operator++() override;

  IndexSet::RangeIterator operator++(int) override;

  reference operator*() const override;

private:
  void fetchNextLeaf();

private:
  llvm::SmallVector<const Node *> nodes;
  const Node *node;
  size_t valueIndex;
};

RTreeIndexSet::RangeIterator::RangeIterator(const Node *root)
    : IndexSet::RangeIterator::Impl(RTree), node(root), valueIndex(0) {
  if (root != nullptr) {
    nodes.push_back(root);

    while (!nodes.back()->isLeaf()) {
      auto current = nodes.pop_back_val();
      const auto &children = current->children;

      for (size_t i = 0, e = children.size(); i < e; ++i) {
        nodes.push_back(children[e - i - 1].get());
      }
    }

    node = nodes.back();
  }
}

std::unique_ptr<IndexSet::RangeIterator::Impl>
RTreeIndexSet::RangeIterator::clone() const {
  return std::make_unique<RTreeIndexSet::RangeIterator>(*this);
}

IndexSet::RangeIterator
RTreeIndexSet::RangeIterator::begin(const RTreeIndexSet &indexSet) {
  return {std::make_unique<RTreeIndexSet::RangeIterator>(indexSet.root.get())};
}

IndexSet::RangeIterator
RTreeIndexSet::RangeIterator::end(const RTreeIndexSet &indexSet) {
  return {std::make_unique<RTreeIndexSet::RangeIterator>(nullptr)};
}

bool RTreeIndexSet::RangeIterator::operator==(
    const IndexSet::RangeIterator::Impl &other) const {
  if (auto *it = other.dyn_cast<RTreeIndexSet::RangeIterator>()) {
    return node == it->node && valueIndex == it->valueIndex;
  }

  return false;
}

bool RTreeIndexSet::RangeIterator::operator!=(
    const IndexSet::RangeIterator::Impl &other) const {
  if (auto *it = other.dyn_cast<RTreeIndexSet::RangeIterator>()) {
    return node != it->node || valueIndex != it->valueIndex;
  }

  return true;
}

IndexSet::RangeIterator::Impl &RTreeIndexSet::RangeIterator::operator++() {
  ++valueIndex;

  if (valueIndex == node->values.size()) {
    fetchNextLeaf();
    valueIndex = 0;
  }

  return *this;
}

IndexSet::RangeIterator RTreeIndexSet::RangeIterator::operator++(int) {
  IndexSet::RangeIterator result(clone());
  ++(*this);
  return result;
}

RTreeIndexSet::RangeIterator::reference
RTreeIndexSet::RangeIterator::operator*() const {
  assert(node != nullptr);
  assert(node->isLeaf());
  return node->values[valueIndex];
}

void RTreeIndexSet::RangeIterator::fetchNextLeaf() {
  nodes.pop_back();

  while (!nodes.empty() && !nodes.back()->isLeaf()) {
    auto current = nodes.pop_back_val();
    const auto &children = current->children;

    for (size_t i = 0, e = children.size(); i < e; ++i) {
      nodes.push_back(children[e - i - 1].get());
    }
  }

  if (nodes.empty()) {
    node = nullptr;
  } else {
    node = nodes.back();
  }
}
} // namespace marco::modeling::impl

//===----------------------------------------------------------------------===//
// R-Tree-IndexSet
//===----------------------------------------------------------------------===//

namespace marco::modeling::impl {
RTreeIndexSet::RTreeIndexSet() : RTreeIndexSet(4, 16) {}

RTreeIndexSet::RTreeIndexSet(size_t minElements, size_t maxElements)
    : IndexSet::Impl(RTree), minElements(minElements), maxElements(maxElements),
      root(nullptr), initialized(false), allowedRank(0) {
  assert(maxElements > 0);
  assert(minElements <= maxElements / 2);
}

RTreeIndexSet::RTreeIndexSet(llvm::ArrayRef<Point> points) : RTreeIndexSet() {
  llvm::SmallVector<MultidimensionalRange> ranges;

  for (const Point &point : points) {
    ranges.emplace_back(point);
  }

  setFromRanges(ranges);
}

RTreeIndexSet::RTreeIndexSet(llvm::ArrayRef<MultidimensionalRange> ranges)
    : RTreeIndexSet() {
  setFromRanges(ranges);
}

RTreeIndexSet::RTreeIndexSet(const RTreeIndexSet &other)
    : IndexSet::Impl(RTree), minElements(other.minElements),
      maxElements(other.maxElements),
      root(other.root == nullptr ? nullptr
                                 : std::make_unique<Node>(*other.root)),
      initialized(other.initialized), allowedRank(other.allowedRank) {}

RTreeIndexSet::RTreeIndexSet(RTreeIndexSet &&other) = default;

RTreeIndexSet::~RTreeIndexSet() = default;

std::unique_ptr<IndexSet::Impl> RTreeIndexSet::clone() const {
  return std::make_unique<RTreeIndexSet>(*this);
}

llvm::raw_ostream &RTreeIndexSet::dump(llvm::raw_ostream &os) const {
  os << "{";

  bool separator = false;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    if (separator) {
      os << ", ";
    }

    separator = true;
    os << range;
  }

  return os << "}";
}

bool RTreeIndexSet::operator==(const Point &rhs) const {
  return *this == RTreeIndexSet(rhs);
}

bool RTreeIndexSet::operator==(const MultidimensionalRange &rhs) const {
  return *this == RTreeIndexSet(rhs);
}

bool RTreeIndexSet::operator==(const IndexSet::Impl &rhs) const {
  if (auto *rhsCasted = rhs.dyn_cast<RTreeIndexSet>()) {
    return *this == *rhsCasted;
  }

  if (rank() != rhs.rank()) {
    return false;
  }

  if (!contains(rhs)) {
    return false;
  }

  return std::all_of(
      rangesBegin(), rangesEnd(),
      [&](const MultidimensionalRange &range) { return rhs.contains(range); });
}

bool RTreeIndexSet::operator==(const RTreeIndexSet &rhs) const {
  if (rank() != rhs.rank()) {
    return false;
  }

  return contains(rhs) &&
         rhs.contains(static_cast<const IndexSet::Impl &>(*this));
}

bool RTreeIndexSet::operator!=(const Point &rhs) const {
  return !(*this == rhs);
}

bool RTreeIndexSet::operator!=(const MultidimensionalRange &rhs) const {
  return !(*this == rhs);
}

bool RTreeIndexSet::operator!=(const IndexSet::Impl &rhs) const {
  return !(*this == rhs);
}

bool RTreeIndexSet::operator!=(const RTreeIndexSet &rhs) const {
  return !(*this == rhs);
}

IndexSet::Impl &RTreeIndexSet::operator+=(const Point &rhs) {
  llvm::SmallVector<Range, 3> ranges;

  for (const Point::data_type &index : rhs) {
    ranges.emplace_back(index, index + 1);
  }

  return *this += MultidimensionalRange(ranges);
}

IndexSet::Impl &RTreeIndexSet::operator+=(const MultidimensionalRange &rhs) {
  if (!initialized) {
    allowedRank = rhs.rank();
    initialized = true;
  }

  assert(rhs.rank() == allowedRank && "Incompatible rank");

  // We must add only the non-existing points.
  llvm::SmallVector<MultidimensionalRange> nonOverlappingRanges;

  if (root == nullptr) {
    nonOverlappingRanges.push_back(rhs);
  } else {
    llvm::SmallVector<MultidimensionalRange> current;
    current.push_back(rhs);

    for (const MultidimensionalRange &range :
         llvm::make_range(rangesBegin(), rangesEnd())) {
      llvm::SmallVector<MultidimensionalRange> next;

      for (const MultidimensionalRange &curr : current) {
        for (MultidimensionalRange &diff : curr.subtract(range)) {
          if (overlaps(diff)) {
            next.push_back(std::move(diff));
          } else {
            // For safety, also check that all the ranges we are going to add
            // do belong to the original range.
            assert(rhs.contains(diff));
            nonOverlappingRanges.push_back(std::move(diff));
          }
        }
      }

      current = std::move(next);
    }
  }

  // The amount of ranges to be inserted may increase due to reinsertions.
  // For this reason, the vector size is computed at each iteration.
  for (size_t i = 0; i < nonOverlappingRanges.size(); ++i) {
    MultidimensionalRange &range = nonOverlappingRanges[i];

    // Check that all the range do not overlap the existing points.
    assert(!overlaps(range));

    if (root == nullptr) {
      root = std::make_unique<Node>(nullptr, range);
      root->add(std::move(range));
    } else {
      // Find position for the new record.
      Node *node = chooseLeaf(range);

      // Add the record to the leaf node.
      node->add(std::move(range));

      // Merge the adjacent ranges.
      llvm::sort(node->values);
      merge(node->values);

      // Split and collapse nodes, if necessary.
      adjustTree(node, nonOverlappingRanges);
    }

    // Check that all the invariants are respected.
    assert(isValid());
  }

  return *this;
}

IndexSet::Impl &RTreeIndexSet::operator+=(const IndexSet::Impl &rhs) {
  if (auto *rhsCasted = rhs.dyn_cast<RTreeIndexSet>()) {
    return *this += *rhsCasted;
  }

  for (const MultidimensionalRange &range :
       llvm::make_range(rhs.rangesBegin(), rhs.rangesEnd())) {
    *this += range;
  }

  return *this;
}

IndexSet::Impl &RTreeIndexSet::operator+=(const RTreeIndexSet &rhs) {
  for (const MultidimensionalRange &range :
       llvm::make_range(rhs.rangesBegin(), rhs.rangesEnd())) {
    *this += range;
  }

  return *this;
}

IndexSet::Impl &RTreeIndexSet::operator-=(const Point &rhs) {
  llvm::SmallVector<Range, 3> ranges;

  for (const Point::data_type &index : rhs) {
    ranges.emplace_back(index, index + 1);
  }

  return *this -= MultidimensionalRange(ranges);
}

IndexSet::Impl &RTreeIndexSet::operator-=(const MultidimensionalRange &rhs) {
  if (!initialized) {
    allowedRank = rhs.rank();
    initialized = true;
  }

  assert(rhs.rank() == allowedRank && "Incompatible rank");

  if (empty()) {
    return *this;
  }

  llvm::SmallVector<MultidimensionalRange> toInsert;
  bool modified;

  do {
    modified = false;
    llvm::SmallVector<Node *> nodeVisits;

    if (root->getBoundary().overlaps(rhs)) {
      nodeVisits.push_back(root.get());
    }

    while (!nodeVisits.empty()) {
      Node *node = nodeVisits.pop_back_val();

      if (!node->isLeaf()) {
        for (auto &child : node->children) {
          if (child->getBoundary().overlaps(rhs)) {
            nodeVisits.push_back(child.get());
          }
        }

        continue;
      }

      llvm::SmallVector<MultidimensionalRange> differences;

      for (MultidimensionalRange &value : node->values) {
        if (value.overlaps(rhs)) {
          modified = true;
          llvm::append_range(differences, value.subtract(rhs));
        } else {
          differences.emplace_back(std::move(value));
        }
      }

      node->values = std::move(differences);

      if (modified) {
        // Split and collapse nodes, if necessary.
        adjustTree(node, toInsert);
        break;
      }
    }
  } while (modified && root);

  for (const MultidimensionalRange &range : toInsert) {
    *this += range;
  }

  return *this;
}

IndexSet::Impl &RTreeIndexSet::operator-=(const IndexSet::Impl &rhs) {
  if (auto *rhsCasted = rhs.dyn_cast<RTreeIndexSet>()) {
    return *this -= *rhsCasted;
  }

  for (const MultidimensionalRange &range :
       llvm::make_range(rhs.rangesBegin(), rhs.rangesEnd())) {
    *this -= range;
  }

  return *this;
}

IndexSet::Impl &RTreeIndexSet::operator-=(const RTreeIndexSet &other) {
  if (empty() || other.empty()) {
    return *this;
  }

  using OverlappingNodePair = std::pair<const Node *, const Node *>;
  llvm::SmallVector<OverlappingNodePair> overlappingNodePairs;

  llvm::SmallVector<std::reference_wrapper<const MultidimensionalRange>>
      overlappingRanges;

  const Node *lhsRoot = root.get();
  const Node *rhsRoot = other.root.get();

  if (lhsRoot->getBoundary().overlaps(rhsRoot->getBoundary())) {
    overlappingNodePairs.emplace_back(lhsRoot, rhsRoot);
  }

  while (!overlappingNodePairs.empty()) {
    OverlappingNodePair overlappingNodePair =
        overlappingNodePairs.pop_back_val();

    const Node *lhs = overlappingNodePair.first;
    const Node *rhs = overlappingNodePair.second;

    if (lhs->isLeaf()) {
      if (rhs->isLeaf()) {
        for (const MultidimensionalRange &lhsRange : lhs->values) {
          for (const MultidimensionalRange &rhsRange : rhs->values) {
            if (lhsRange.overlaps(rhsRange)) {
              overlappingRanges.emplace_back(rhsRange);
            }
          }
        }
      } else {
        for (const auto &child : rhs->children) {
          if (lhs->getBoundary().overlaps(child->getBoundary())) {
            overlappingNodePairs.emplace_back(lhs, child.get());
          }
        }
      }
    } else {
      for (const auto &child : lhs->children) {
        if (child->getBoundary().overlaps(rhs->getBoundary())) {
          overlappingNodePairs.emplace_back(child.get(), rhs);
        }
      }
    }
  }

  for (const auto &overlappingRange : overlappingRanges) {
    *this -= overlappingRange;
  }

  return *this;
}

bool RTreeIndexSet::empty() const { return root == nullptr; }

size_t RTreeIndexSet::rank() const { return allowedRank; }

size_t RTreeIndexSet::flatSize() const {
  size_t result = 0;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    result += range.flatSize();
  }

  return result;
}

void RTreeIndexSet::clear() { root = nullptr; }

RTreeIndexSet::const_point_iterator RTreeIndexSet::begin() const {
  return RTreeIndexSet::PointIterator::begin(*this);
}

RTreeIndexSet::const_point_iterator RTreeIndexSet::end() const {
  return RTreeIndexSet::PointIterator::end(*this);
}

RTreeIndexSet::const_range_iterator RTreeIndexSet::rangesBegin() const {
  return RTreeIndexSet::RangeIterator::begin(*this);
}

RTreeIndexSet::const_range_iterator RTreeIndexSet::rangesEnd() const {
  return RTreeIndexSet::RangeIterator::end(*this);
}

bool RTreeIndexSet::contains(const Point &other) const {
  return contains(MultidimensionalRange(other));
}

bool RTreeIndexSet::contains(const MultidimensionalRange &other) const {
  if (root == nullptr) {
    return false;
  }

  std::queue<MultidimensionalRange> remainingRanges;
  remainingRanges.push(other);
  bool changesDetected = true;

  while (!remainingRanges.empty() && changesDetected) {
    changesDetected = false;
    const MultidimensionalRange &range = remainingRanges.front();

    llvm::SmallVector<const Node *> nodes;

    if (root->getBoundary().overlaps(range)) {
      nodes.push_back(root.get());
    }

    while (!nodes.empty() && !changesDetected) {
      auto node = nodes.pop_back_val();

      for (const auto &child : node->children) {
        if (child->getBoundary().overlaps(range)) {
          nodes.push_back(child.get());
        }
      }

      for (const auto &value : node->values) {
        if (value.overlaps(range)) {
          for (MultidimensionalRange &difference : range.subtract(value)) {
            remainingRanges.push(std::move(difference));
          }

          remainingRanges.pop();
          changesDetected = true;
          break;
        }
      }
    }
  }

  return remainingRanges.empty();
}

bool RTreeIndexSet::contains(const IndexSet::Impl &other) const {
  if (auto *otherCasted = other.dyn_cast<RTreeIndexSet>()) {
    bool optimizedResult = contains(*otherCasted);
    assert(optimizedResult == containsGenericIndexSet(other));
    return optimizedResult;
  }

  return containsGenericIndexSet(other);
}

bool RTreeIndexSet::contains(const RTreeIndexSet &other) const {
  if (other.empty()) {
    return true;
  }

  if (empty()) {
    return false;
  }

  const Node *lhsRoot = getRoot();
  const Node *rhsRoot = other.getRoot();

  if (!lhsRoot->getBoundary().contains(rhsRoot->getBoundary())) {
    return false;
  }

  using OverlappingNode = std::pair<const Node *, std::vector<const Node *>>;

  llvm::SmallVector<OverlappingNode> overlappingNodes;

  overlappingNodes.emplace_back(rhsRoot, std::vector<const Node *>({lhsRoot}));

  while (!overlappingNodes.empty()) {
    OverlappingNode overlappingNode = overlappingNodes.pop_back_val();

    const Node *rhs = overlappingNode.first;
    const auto &lhsNodes = overlappingNode.second;

    if (rhs->isLeaf()) {
      if (llvm::any_of(lhsNodes,
                       [](const Node *node) { return !node->isLeaf(); })) {
        std::vector<const Node *> newLhsNodes;

        for (const Node *lhs : lhsNodes) {
          if (lhs->isLeaf()) {
            newLhsNodes.push_back(lhs);
          } else {
            for (const auto &child : lhs->children) {
              if (child->getBoundary().overlaps(rhs->getBoundary())) {
                newLhsNodes.push_back(child.get());
              }
            }
          }
        }

        overlappingNodes.emplace_back(rhs, newLhsNodes);
      } else {
        for (const MultidimensionalRange &value : rhs->values) {
          llvm::SmallVector<MultidimensionalRange, 3> remainingRanges;
          remainingRanges.push_back(value);

          for (const auto &lhs : lhsNodes) {
            assert(lhs->isLeaf());

            for (const MultidimensionalRange &lhsRange : lhs->values) {
              llvm::SmallVector<MultidimensionalRange, 3> newRemaining;

              for (const auto &remainingRange : remainingRanges) {
                for (auto &diff : remainingRange.subtract(lhsRange)) {
                  newRemaining.push_back(std::move(diff));
                }
              }

              remainingRanges = std::move(newRemaining);
            }
          }

          if (!remainingRanges.empty()) {
            return false;
          }
        }
      }
    } else {
      for (const auto &child : rhs->children) {
        bool anyOverlap = false;
        std::vector<const Node *> childOverlappingNodes;

        for (const auto &lhs : lhsNodes) {
          if (lhs->isLeaf()) {
            if (lhs->getBoundary().overlaps(child->getBoundary())) {
              childOverlappingNodes.push_back(lhs);
              anyOverlap = true;
            }
          } else {
            for (const auto &lhsChild : lhs->children) {
              if (lhsChild->getBoundary().overlaps(child->getBoundary())) {
                childOverlappingNodes.push_back(lhs);
                anyOverlap = true;
              }
            }
          }
        }

        if (!anyOverlap) {
          return false;
        }

        overlappingNodes.emplace_back(child.get(), childOverlappingNodes);
      }
    }
  }

  return true;
}

bool RTreeIndexSet::containsGenericIndexSet(const IndexSet::Impl &other) const {
  return std::all_of(
      other.rangesBegin(), other.rangesEnd(),
      [&](const MultidimensionalRange &range) { return contains(range); });
}

bool RTreeIndexSet::overlaps(const MultidimensionalRange &other) const {
  if (root == nullptr) {
    return false;
  }

  llvm::SmallVector<const Node *> nodes;

  if (root->getBoundary().overlaps(other)) {
    nodes.push_back(root.get());
  }

  while (!nodes.empty()) {
    auto node = nodes.pop_back_val();

    for (const auto &child : node->children) {
      if (child->getBoundary().overlaps(other)) {
        nodes.push_back(child.get());
      }
    }

    for (const auto &value : node->values) {
      if (value.overlaps(other)) {
        return true;
      }
    }
  }

  return false;
}

bool RTreeIndexSet::overlaps(const IndexSet::Impl &other) const {
  if (auto *otherCasted = other.dyn_cast<RTreeIndexSet>()) {
    bool optimizedResult = overlaps(*otherCasted);
    assert(optimizedResult == overlapsGenericIndexSet(other));
    return optimizedResult;
  }

  return overlapsGenericIndexSet(other);
}

bool RTreeIndexSet::overlaps(const RTreeIndexSet &other) const {
  if (empty() || other.empty()) {
    return false;
  }

  using OverlappingNode = std::pair<const Node *, const Node *>;

  llvm::SmallVector<OverlappingNode> overlappingNodes;

  const Node *lhsRoot = getRoot();
  const Node *rhsRoot = other.getRoot();

  if (lhsRoot->getBoundary().overlaps(rhsRoot->getBoundary())) {
    overlappingNodes.emplace_back(lhsRoot, rhsRoot);
  }

  while (!overlappingNodes.empty()) {
    OverlappingNode overlappingNode = overlappingNodes.pop_back_val();

    const Node *lhs = overlappingNode.first;
    const Node *rhs = overlappingNode.second;

    if (lhs->isLeaf()) {
      if (rhs->isLeaf()) {
        for (const MultidimensionalRange &lhsRange : lhs->values) {
          for (const MultidimensionalRange &rhsRange : rhs->values) {
            if (lhsRange.overlaps(rhsRange)) {
              return true;
            }
          }
        }
      } else {
        for (const auto &child : rhs->children) {
          if (child->getBoundary().overlaps(lhs->getBoundary())) {
            overlappingNodes.emplace_back(lhs, child.get());
          }
        }
      }
    } else {
      for (const auto &child : lhs->children) {
        if (child->getBoundary().overlaps(rhs->getBoundary())) {
          overlappingNodes.emplace_back(child.get(), rhs);
        }
      }
    }
  }

  return false;
}

bool RTreeIndexSet::overlapsGenericIndexSet(const IndexSet::Impl &other) const {
  return llvm::any_of(
      llvm::make_range(other.rangesBegin(), other.rangesEnd()),
      [&](const MultidimensionalRange &range) { return overlaps(range); });
}

IndexSet RTreeIndexSet::intersect(const MultidimensionalRange &other) const {
  if (root == nullptr) {
    return {};
  }

  llvm::SmallVector<const Node *> nodes;

  if (root->getBoundary().overlaps(other)) {
    nodes.push_back(root.get());
  } else {
    return {};
  }

  llvm::SmallVector<MultidimensionalRange> result;

  while (!nodes.empty()) {
    const Node *node = nodes.pop_back_val();

    for (const auto &child : node->children) {
      if (child->getBoundary().overlaps(other)) {
        nodes.push_back(child.get());
      }
    }

    for (const auto &value : node->values) {
      if (value.overlaps(other)) {
        result.push_back(other.intersect(value));
      }
    }
  }

  return {result};
}

IndexSet RTreeIndexSet::intersect(const IndexSet::Impl &other) const {
  if (auto *otherCasted = other.dyn_cast<RTreeIndexSet>()) {
    return intersect(*otherCasted);
  }

  IndexSet result;

  if (root == nullptr || other.empty()) {
    return result;
  }

  for (const MultidimensionalRange &range :
       llvm::make_range(other.rangesBegin(), other.rangesEnd())) {
    result += this->intersect(range);
  }

  return result;
}

IndexSet RTreeIndexSet::intersect(const RTreeIndexSet &other) const {
  if (empty() || other.empty()) {
    return {};
  }

  using OverlappingNodePair = std::pair<const Node *, const Node *>;
  llvm::SmallVector<OverlappingNodePair> overlappingNodes;

  const Node *lhsRoot = getRoot();
  const Node *rhsRoot = other.getRoot();

  if (lhsRoot->getBoundary().overlaps(rhsRoot->getBoundary())) {
    overlappingNodes.emplace_back(lhsRoot, rhsRoot);
  }

  llvm::SmallVector<MultidimensionalRange> result;

  while (!overlappingNodes.empty()) {
    OverlappingNodePair overlappingNodePair = overlappingNodes.pop_back_val();

    const Node *lhs = overlappingNodePair.first;
    const Node *rhs = overlappingNodePair.second;

    if (lhs->isLeaf()) {
      if (rhs->isLeaf()) {
        for (const MultidimensionalRange &lhsRange : lhs->values) {
          for (const MultidimensionalRange &rhsRange : rhs->values) {
            if (lhsRange.overlaps(rhsRange)) {
              result.push_back(lhsRange.intersect(rhsRange));
            }
          }
        }
      } else {
        for (const auto &child : rhs->children) {
          if (child->getBoundary().overlaps(lhs->getBoundary())) {
            overlappingNodes.emplace_back(lhs, child.get());
          }
        }
      }
    } else {
      for (const auto &child : lhs->children) {
        if (child->getBoundary().overlaps(rhs->getBoundary())) {
          overlappingNodes.emplace_back(child.get(), rhs);
        }
      }
    }
  }

  return {result};
}

IndexSet RTreeIndexSet::complement(const MultidimensionalRange &other) const {
  if (root == nullptr) {
    return {other};
  }

  std::vector<MultidimensionalRange> result;

  std::vector<MultidimensionalRange> current;
  current.push_back(other);

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    std::vector<MultidimensionalRange> next;

    for (const MultidimensionalRange &curr : current) {
      for (MultidimensionalRange &diff : curr.subtract(range)) {
        if (overlaps(diff)) {
          next.push_back(std::move(diff));
        } else {
          result.push_back(std::move(diff));
        }
      }
    }

    current = next;
  }

  return {result};
}

std::unique_ptr<IndexSet::Impl>
RTreeIndexSet::slice(const llvm::BitVector &filter) const {
  if (!initialized) {
    return clone();
  }

  llvm::SmallVector<MultidimensionalRange> result;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    result.push_back(range.slice(filter));
  }

  return std::make_unique<RTreeIndexSet>(std::move(result));
}

std::unique_ptr<IndexSet::Impl>
RTreeIndexSet::takeFirstDimensions(size_t n) const {
  if (root == nullptr) {
    return clone();
  }

  assert(n <= rank());
  llvm::SmallVector<MultidimensionalRange> result;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    result.push_back(range.takeFirstDimensions(n));
  }

  return std::make_unique<RTreeIndexSet>(std::move(result));
}

std::unique_ptr<IndexSet::Impl>
RTreeIndexSet::takeLastDimensions(size_t n) const {
  if (root == nullptr) {
    return clone();
  }

  assert(n <= rank());
  llvm::SmallVector<MultidimensionalRange> result;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    result.push_back(range.takeLastDimensions(n));
  }

  return std::make_unique<RTreeIndexSet>(std::move(result));
}

std::unique_ptr<IndexSet::Impl>
RTreeIndexSet::takeDimensions(const llvm::SmallBitVector &dimensions) const {
  assert(dimensions.size() == rank());
  llvm::SmallVector<MultidimensionalRange> result;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    result.push_back(range.takeDimensions(dimensions));
  }

  return std::make_unique<RTreeIndexSet>(std::move(result));
}

std::unique_ptr<IndexSet::Impl>
RTreeIndexSet::dropFirstDimensions(size_t n) const {
  assert(n < rank());
  llvm::SmallVector<MultidimensionalRange> result;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    result.push_back(range.dropFirstDimensions(n));
  }

  return std::make_unique<RTreeIndexSet>(std::move(result));
}

std::unique_ptr<IndexSet::Impl>
RTreeIndexSet::dropLastDimensions(size_t n) const {
  assert(n < rank());
  llvm::SmallVector<MultidimensionalRange> result;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    result.push_back(range.dropLastDimensions(n));
  }

  return std::make_unique<RTreeIndexSet>(std::move(result));
}

std::unique_ptr<IndexSet::Impl>
RTreeIndexSet::prepend(const IndexSet &other) const {
  llvm::SmallVector<MultidimensionalRange> result;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    for (const MultidimensionalRange &otherRange :
         llvm::make_range(other.rangesBegin(), other.rangesEnd())) {
      result.push_back(range.prepend(otherRange));
    }
  }

  return std::make_unique<RTreeIndexSet>(std::move(result));
}

std::unique_ptr<IndexSet::Impl>
RTreeIndexSet::append(const IndexSet &other) const {
  llvm::SmallVector<MultidimensionalRange> result;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    for (const MultidimensionalRange &otherRange :
         llvm::make_range(other.rangesBegin(), other.rangesEnd())) {
      result.push_back(range.append(otherRange));
    }
  }

  return std::make_unique<RTreeIndexSet>(std::move(result));
}

std::unique_ptr<IndexSet::Impl>
RTreeIndexSet::getCanonicalRepresentation() const {
  // Collect all the ranges in advance, so that the canonicalization has to
  // be performed only once, and not every time a new range is inserted.
  llvm::SmallVector<MultidimensionalRange> ranges;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    ranges.push_back(range);
  }

  return std::make_unique<ListIndexSet>(std::move(ranges));
}

const RTreeIndexSet::Node *RTreeIndexSet::getRoot() const { return root.get(); }

void RTreeIndexSet::setFromRanges(
    llvm::ArrayRef<MultidimensionalRange> ranges) {
  root = nullptr;

  if (ranges.empty()) {
    return;
  }

  initialized = true;
  allowedRank = ranges.front().rank();

  // Sort the original ranges.
  llvm::SmallVector<MultidimensionalRange> sorted;
  llvm::append_range(sorted, ranges);
  llvm::sort(sorted);
  merge(sorted);

  // Ensure that the ranges don't overlap.
  llvm::SmallVector<MultidimensionalRange> nonOverlappingRanges;

  for (const MultidimensionalRange &current : sorted) {
    if (nonOverlappingRanges.empty()) {
      nonOverlappingRanges.push_back(current);
      continue;
    }

    size_t possibleOverlapIndex = nonOverlappingRanges.size();

    while (possibleOverlapIndex > 0 &&
           current.anyDimensionOverlaps(
               nonOverlappingRanges[possibleOverlapIndex - 1])) {
      --possibleOverlapIndex;
    }

    llvm::SmallVector<MultidimensionalRange> differences;
    differences.push_back(current);

    for (size_t i = possibleOverlapIndex, e = nonOverlappingRanges.size();
         i < e; ++i) {
      const MultidimensionalRange &other = nonOverlappingRanges[i];
      llvm::SmallVector<MultidimensionalRange> newDifferences;

      for (MultidimensionalRange &currentDifference : differences) {
        if (currentDifference.overlaps(other)) {
          llvm::append_range(newDifferences, currentDifference.subtract(other));
        } else {
          newDifferences.push_back(std::move(currentDifference));
        }
      }

      differences = std::move(newDifferences);
    }

    llvm::append_range(nonOverlappingRanges, std::move(differences));

    llvm::sort(
        std::next(std::begin(nonOverlappingRanges), possibleOverlapIndex),
        std::end(nonOverlappingRanges));
  }

  assert(llvm::none_of(nonOverlappingRanges,
                       [&](const MultidimensionalRange &range) {
                         for (const MultidimensionalRange &other :
                              nonOverlappingRanges) {
                           if (range.overlaps(other) && range != other) {
                             return true;
                           }
                         }

                         return false;
                       }) &&
         "Overlapping ranges found during initialization of R-Tree IndexSet");

  // Create the tree structure.
  llvm::SmallVector<std::unique_ptr<Node>> nodes;

  if (nonOverlappingRanges.size() > maxElements) {
    size_t numOfGroups = nonOverlappingRanges.size() / minElements;
    size_t remainingElements = nonOverlappingRanges.size() % minElements;

    for (size_t i = 0; i < numOfGroups; ++i) {
      auto &node = nodes.emplace_back(std::make_unique<Node>(
          nullptr, nonOverlappingRanges[i * minElements]));

      for (size_t j = 0; j < minElements; ++j) {
        node->values.push_back(
            std::move(nonOverlappingRanges[i * minElements + j]));
      }

      node->recalcBoundary();
    }

    for (size_t i = 0; i < remainingElements; ++i) {
      nodes[i]->values.push_back(
          std::move(nonOverlappingRanges[numOfGroups * minElements + i]));
    }

    for (auto &node : nodes) {
      node->recalcBoundary();
    }
  } else {
    auto &node = nodes.emplace_back(
        std::make_unique<Node>(nullptr, nonOverlappingRanges[0]));

    llvm::append_range(node->values, std::move(nonOverlappingRanges));
    node->recalcBoundary();
  }

  while (nodes.size() > maxElements) {
    llvm::SmallVector<std::unique_ptr<Node>> newNodes;

    for (size_t i = 0, e = nodes.size(); i < e; i += maxElements) {
      auto &parent = newNodes.emplace_back(
          std::make_unique<Node>(nullptr, nodes[i]->getBoundary()));

      for (size_t j = 0; j < maxElements; ++j) {
        if (i + j >= e) {
          break;
        }

        auto &child = parent->children.emplace_back(std::move(nodes[i + j]));
        child->parent = parent.get();
      }

      parent->recalcBoundary();
    }

    nodes = std::move(newNodes);
  }

  if (nodes.size() > 1) {
    auto parent = std::make_unique<Node>(nullptr, nodes[0]->getBoundary());

    for (auto &child : nodes) {
      child->parent = parent.get();
      parent->children.emplace_back(std::move(child));
    }

    parent->recalcBoundary();

    nodes.clear();
    nodes.push_back(std::move(parent));
  }

  root = std::move(nodes.front());

  assert(isValid() && "Inconsistent initialization of the R-Tree IndexSet");

  assert(llvm::all_of(ranges,
                      [&](const MultidimensionalRange &range) {
                        for (Point point : range) {
                          if (!contains(point)) {
                            return false;
                          }
                        }

                        return true;
                      }) &&
         "Not all indices have been inserted");

  assert(llvm::all_of(*this,
                      [&](Point point) {
                        for (const MultidimensionalRange &range : ranges) {
                          if (range.contains(point)) {
                            return true;
                          }
                        }

                        return false;
                      }) &&
         "Some non-requested indices have been inserted");
}

RTreeIndexSet::Node *
RTreeIndexSet::chooseLeaf(const MultidimensionalRange &entry) const {
  Node *node = root.get();

  while (!node->isLeaf()) {
    // Choose the child that needs the least enlargement to include the new
    // element.
    std::vector<std::pair<MultidimensionalRange, size_t>> candidateMBRs;

    for (const auto &child : node->children) {
      auto mbr = getMBR(child->getBoundary(), entry);
      size_t flatDifference = mbr.flatSize() - child->getFlatSize();
      candidateMBRs.emplace_back(std::move(mbr), flatDifference);
    }

    // Select the child that needs the least enlargement to include the new
    // entry. Resolve ties by choosing the node with the rectangle with the
    // smallest area.
    auto enumeratedCandidates = llvm::enumerate(candidateMBRs);

    auto it = std::min_element(
        enumeratedCandidates.begin(), enumeratedCandidates.end(),
        [](const auto &first, const auto &second) {
          if (first.value().second == second.value().second) {
            return first.value().first.flatSize() <
                   second.value().first.flatSize();
          }

          return first.value().second < second.value().second;
        });

    node = node->children[(*it).index()].get();
  }

  return node;
}
} // namespace marco::modeling::impl

namespace {
/// Select two entries to be the first elements of the new groups.
template <typename T>
std::pair<size_t, size_t> pickSeeds(llvm::ArrayRef<T> entries) {
  std::vector<std::tuple<size_t, size_t, size_t>> candidates;

  for (size_t i = 0; i < entries.size(); ++i) {
    for (size_t j = i + 1; j < entries.size(); ++j) {
      auto mbr = getMBR(getShape(entries[i]), getShape(entries[j]));

      auto difference = mbr.flatSize() - getShape(entries[i]).flatSize() -
                        getShape(entries[j]).flatSize();

      candidates.emplace_back(i, j, difference);
    }
  }

  auto it = std::max_element(candidates.begin(), candidates.end(),
                             [](const auto &first, const auto &second) {
                               return std::get<2>(first) < std::get<2>(second);
                             });

  return std::make_pair(std::get<0>(*it), std::get<1>(*it));
}

/// Select one remaining entry for classification in a group.
template <typename T>
size_t pickNext(llvm::ArrayRef<T> entries,
                const llvm::DenseSet<size_t> &alreadyProcessedEntries,
                const RTreeIndexSet::Node &firstNode,
                const RTreeIndexSet::Node &secondNode) {
  assert(!entries.empty());

  // Determine the cost of putting each entry in each group.
  std::vector<std::pair<size_t, size_t>> costs;

  for (const auto &entry : llvm::enumerate(entries)) {
    if (alreadyProcessedEntries.contains(entry.index())) {
      continue;
    }

    auto d1 =
        getMBR(getShape(entry.value()), firstNode.getBoundary()).flatSize() -
        firstNode.getBoundary().flatSize();
    auto d2 =
        getMBR(getShape(entry.value()), secondNode.getBoundary()).flatSize() -
        secondNode.getBoundary().flatSize();
    auto minMax = std::minmax(d1, d2);
    costs.emplace_back(entry.index(), minMax.second - minMax.first);
  }

  // Find the entry with the greatest preference for one group.
  assert(!costs.empty());

  auto it = std::max_element(costs.begin(), costs.end(),
                             [](const auto &first, const auto &second) {
                               return first.second < second.second;
                             });

  return it->first;
}
} // namespace

namespace {
template <typename T>
std::pair<std::unique_ptr<RTreeIndexSet::Node>,
          std::unique_ptr<RTreeIndexSet::Node>>
splitNode(RTreeIndexSet::Node &node, const size_t minElements,
          llvm::function_ref<llvm::SmallVectorImpl<T> &(RTreeIndexSet::Node &)>
              containerFn) {
  auto seeds = pickSeeds(llvm::ArrayRef(containerFn(node)));

  llvm::DenseSet<size_t> movedValues;

  auto moveValueFn = [&](RTreeIndexSet::Node &destination, size_t valueIndex) {
    if (!movedValues.contains(valueIndex)) {
      containerFn(destination)
          .push_back(std::move(containerFn(node)[valueIndex]));
      movedValues.insert(valueIndex);
    }
  };

  auto firstNew = std::make_unique<RTreeIndexSet::Node>(
      node.parent, getShape(containerFn(node)[seeds.first]));

  moveValueFn(*firstNew, seeds.first);

  auto secondNew = std::make_unique<RTreeIndexSet::Node>(
      node.parent, getShape(containerFn(node)[seeds.second]));

  moveValueFn(*secondNew, seeds.second);

  // Keep processing the values until all of them have been assigned to the
  // new nodes.
  while (movedValues.size() != containerFn(node).size()) {
    auto remaining = containerFn(node).size() - movedValues.size();

    // If one group has so few entries that all the rest must be assigned to
    // it in order for it to have the minimum number of element, then assign
    // them and stop.

    if (containerFn(*firstNew).size() + remaining == minElements) {
      auto &container = containerFn(node);

      for (size_t i = 0, e = container.size(); i < e; ++i) {
        moveValueFn(*firstNew, i);
      }

      assert(movedValues.size() == containerFn(node).size());
      break;

    } else if (containerFn(*secondNew).size() + remaining == minElements) {
      auto &container = containerFn(node);

      for (size_t i = 0, e = container.size(); i < e; ++i) {
        moveValueFn(*secondNew, i);
      }

      assert(movedValues.size() == containerFn(node).size());
      break;
    }

    // Choose the next entry to assign.
    auto next = pickNext(llvm::ArrayRef(containerFn(node)), movedValues,
                         *firstNew, *secondNew);

    assert(movedValues.find(next) == movedValues.end());

    // Add it to the group whose covering rectangle will have to be
    // enlarged least to accommodate it. Resolve ties by adding the entry to
    // the group with smaller area, then to the one with fewer entries, then
    // to either of them.

    auto firstEnlargement =
        getMBR(getShape(containerFn(node)[next]), firstNew->getBoundary())
            .flatSize() -
        firstNew->getFlatSize();

    auto secondEnlargement =
        getMBR(getShape(containerFn(node)[next]), secondNew->getBoundary())
            .flatSize() -
        secondNew->getFlatSize();

    if (firstEnlargement == secondEnlargement) {
      auto firstArea = firstNew->getFlatSize();
      auto secondArea = secondNew->getFlatSize();

      if (firstArea == secondArea) {
        auto firstElementsAmount = containerFn(*firstNew).size();
        auto secondElementsAmount = containerFn(*secondNew).size();

        if (firstElementsAmount <= secondElementsAmount) {
          moveValueFn(*firstNew, next);
        } else {
          moveValueFn(*secondNew, next);
        }
      } else if (firstArea < secondArea) {
        moveValueFn(*firstNew, next);
      } else {
        moveValueFn(*secondNew, next);
      }
    } else if (firstEnlargement < secondEnlargement) {
      moveValueFn(*firstNew, next);
    } else {
      moveValueFn(*secondNew, next);
    }
  }

  assert(containerFn(*firstNew).size() >= minElements);
  assert(containerFn(*secondNew).size() >= minElements);

  // Update the boundaries of the nodes.
  firstNew->recalcBoundary();
  secondNew->recalcBoundary();

  return std::make_pair(std::move(firstNew), std::move(secondNew));
}
} // namespace

namespace marco::modeling {
std::pair<std::unique_ptr<RTreeIndexSet::Node>,
          std::unique_ptr<RTreeIndexSet::Node>>
RTreeIndexSet::splitNode(Node &node) {
  if (node.isLeaf()) {
    return ::splitNode<MultidimensionalRange>(
        node, minElements,
        [](Node &node) -> llvm::SmallVectorImpl<MultidimensionalRange> & {
          return node.values;
        });
  } else {
    auto result = ::splitNode<std::unique_ptr<Node>>(
        node, minElements,
        [](Node &node) -> llvm::SmallVectorImpl<std::unique_ptr<Node>> & {
          return node.children;
        });

    // We need to fix the parent address.
    for (auto &child : result.first->children) {
      child->parent = result.first.get();
    }

    for (auto &child : result.second->children) {
      child->parent = result.second.get();
    }

    return result;
  }
}

void RTreeIndexSet::adjustTree(
    Node *node, llvm::SmallVectorImpl<MultidimensionalRange> &toReinsert) {
  if (node->isRoot() && node->fanOut() == 0) {
    root = nullptr;
    node = nullptr;
  } else if (!node->isRoot() && node->fanOut() < minElements) {
    while (node && !node->isRoot() && node->fanOut() < minElements) {
      llvm::SmallVector<Node *> nodes;
      nodes.push_back(node);

      while (!nodes.empty()) {
        auto current = nodes.pop_back_val();

        for (auto &child : current->children) {
          nodes.push_back(child.get());
        }

        for (auto &value : current->values) {
          assert(value.rank() == rank());
          toReinsert.push_back(std::move(value));
        }
      }

      Node *parent = node->parent;
      assert(parent != nullptr);

      // The new children of the parent will be the old ones except the
      // current node, which has been collapsed and its ranges queued for
      // a new insertion.
      llvm::SmallVector<std::unique_ptr<Node>> newChildren;

      for (auto &child : node->parent->children) {
        if (child.get() != node) {
          newChildren.push_back(std::move(child));
        }
      }

      parent->children = std::move(newChildren);

      if (parent->children.empty()) {
        // If the root has no children, then the tree has to be
        // reinitialized.
        assert(parent->isRoot());
        root = nullptr;
        node = nullptr;
      } else {
        parent->recalcBoundary();
        node = parent;
      }
    }
  } else if (node->fanOut() > maxElements) {
    // Propagate node splits.
    while (node->fanOut() > maxElements) {
      auto newNodes = splitNode(*node);

      if (node->isRoot()) {
        // If node split propagation caused the root to split, then
        // create a new root whose children are the two resulting nodes.

        auto rootBoundary = getMBR(newNodes.first->getBoundary(),
                                   newNodes.second->getBoundary());

        root = std::make_unique<Node>(nullptr, rootBoundary);

        newNodes.first->parent = root.get();
        newNodes.second->parent = root.get();

        root->add(std::move(newNodes.first));
        root->add(std::move(newNodes.second));

        node = root.get();
        break;
      }

      *node = std::move(*newNodes.first);

      for (auto &child : node->children) {
        child->parent = node;
      }

      node->parent->add(std::move(newNodes.second));

      // Propagate changes upward.
      node = node->parent;
    }
  }

  // Fix all the boundaries up to the root.
  while (node != nullptr) {
    node->recalcBoundary();
    node = node->parent;
  }
}

} // namespace marco::modeling

namespace {
/// Check that all the children of a node has the correct parent set.
bool checkParentRelationships(const impl::RTreeIndexSet &indexSet) {
  llvm::SmallVector<const impl::RTreeIndexSet::Node *> nodes;

  if (auto root = indexSet.getRoot(); root != nullptr) {
    nodes.push_back(root);
  }

  while (!nodes.empty()) {
    auto node = nodes.pop_back_val();

    for (const auto &child : node->children) {
      if (child->parent != node) {
        return false;
      }

      nodes.push_back(child.get());
    }
  }

  return true;
}

/// Check the correctness of the MBR of all the nodes.
bool checkMBRsInvariant(const impl::RTreeIndexSet &indexSet) {
  llvm::SmallVector<const impl::RTreeIndexSet::Node *> nodes;

  if (auto root = indexSet.getRoot(); root != nullptr) {
    nodes.push_back(root);
  }

  while (!nodes.empty()) {
    auto node = nodes.pop_back_val();

    if (node->isLeaf()) {
      if (node->getBoundary() != getMBR(llvm::ArrayRef(node->values))) {
        return false;
      }
    } else {
      if (node->getBoundary() != getMBR(llvm::ArrayRef(node->children))) {
        return false;
      }
    }

    for (const auto &child : node->children) {
      nodes.push_back(child.get());
    }
  }

  return true;
}

/// Check that all the nodes have between a minimum and a maximum
/// amount of out edges (apart the root node).
bool checkFanOutInvariant(const impl::RTreeIndexSet &indexSet) {
  llvm::SmallVector<const impl::RTreeIndexSet::Node *> nodes;

  if (auto root = indexSet.getRoot(); root != nullptr) {
    for (const auto &child : root->children) {
      nodes.push_back(child.get());
    }
  }

  while (!nodes.empty()) {
    auto node = nodes.pop_back_val();

    if (size_t size = node->fanOut();
        size < indexSet.minElements || size > indexSet.maxElements) {
      return false;
    }

    for (const auto &child : node->children) {
      nodes.push_back(child.get());
    }
  }

  return true;
}
} // namespace

namespace marco::modeling::impl {
bool RTreeIndexSet::isValid() const {
  if (!checkParentRelationships(*this)) {
    llvm::dbgs() << "IndexSet: parents hierarchy invariant failure\n";
    return false;
  }

  if (!checkMBRsInvariant(*this)) {
    llvm::dbgs() << "IndexSet: MBRs invariant failure\n";
    return false;
  }

  if (!checkFanOutInvariant(*this)) {
    llvm::dbgs() << "IndexSet: fan-out invariant failure\n";
    return false;
  }

  return true;
}
} // namespace marco::modeling::impl
