#include "marco/Modeling/IndexSet.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include <list>
#include <queue>

using namespace ::marco::modeling;

namespace {
template <typename>
struct Merger {};

template <>
struct Merger<MultidimensionalRange> {
  template <typename It>
  static auto getMergePossibility(const It &it1, const It &it2) {
    return it1->canBeMerged(*it2);
  }

  template <typename It>
  static void merge(It &it1, It &it2, size_t dimension) {
    *it1 = it1->merge(*it2, dimension);
  }
};

template <>
struct Merger<r_tree::impl::Object<MultidimensionalRange>> {
  template <typename It>
  static auto getMergePossibility(const It &it1, const It &it2) {
    return (**it1).canBeMerged(**it2);
  }

  template <typename It>
  static void merge(It &it1, It &it2, size_t dimension) {
    **it1 = (*it1)->merge(**it2, dimension);
  }
};

template <typename T>
void merge(llvm::SmallVectorImpl<T> &ranges) {
  if (ranges.empty()) {
    return;
  }

  using It = typename llvm::SmallVectorImpl<T>::iterator;

  auto findCandidates = [&](It begin, It end) -> std::tuple<It, It, size_t> {
    for (It it1 = begin; it1 != end; ++it1) {
      for (It it2 = std::next(it1); it2 != end; ++it2) {
        if (auto mergePossibility = Merger<T>::getMergePossibility(it1, it2);
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
    Merger<T>::merge(first, second, dimension);
    ranges.erase(second);
    candidates = findCandidates(ranges.begin(), ranges.end());
  }

  assert(!ranges.empty());
}
} // namespace

namespace marco::modeling {
IndexSet::IndexSet(llvm::ArrayRef<Point> points) {
  llvm::SmallVector<MultidimensionalRange> ranges;

  for (const Point &point : points) {
    ranges.emplace_back(point);
  }

  setFromRanges(ranges);
}

IndexSet::IndexSet(llvm::ArrayRef<MultidimensionalRange> ranges) {
  setFromRanges(ranges);
}

bool IndexSet::operator==(const Point &other) const {
  return *this == MultidimensionalRange(other);
}

bool IndexSet::operator==(const MultidimensionalRange &other) const {
  return *this == IndexSet(other);
}

bool IndexSet::operator!=(const Point &other) const {
  return !(*this == other);
}

bool IndexSet::operator!=(const MultidimensionalRange &other) const {
  return !(*this == other);
}

llvm::hash_code hash_value(const IndexSet &value) {
  if (value.empty()) {
    return 0;
  }

  return llvm::hash_value(value.flatSize());
}

int IndexSet::compare(const IndexSet &other) const {
  auto firstIt = rangesBegin();
  auto firstEndIt = rangesEnd();

  auto secondIt = other.rangesBegin();
  auto secondEndIt = other.rangesEnd();

  while (firstIt != firstEndIt && secondIt != secondEndIt) {
    if (auto rangeCmp = (*firstIt).compare(*secondIt); rangeCmp != 0) {
      return rangeCmp;
    }

    ++firstIt;
    ++secondIt;
  }

  if (firstIt == firstEndIt && secondIt != secondEndIt) {
    // The first set has fewer ranges.
    return -1;
  }

  if (firstIt != firstEndIt && secondIt == secondEndIt) {
    // The second set has fewer ranges.
    return 1;
  }

  assert(firstIt == firstEndIt && secondIt == secondEndIt);
  return 0;
}

void IndexSet::setFromRanges(llvm::ArrayRef<MultidimensionalRange> ranges) {
  // Sort the original ranges.
  llvm::SmallVector<MultidimensionalRange> sorted;
  llvm::append_range(sorted, ranges);
  llvm::sort(sorted);

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

  setFromObjects(nonOverlappingRanges);

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

IndexSet &IndexSet::operator+=(Point other) {
  return *this += MultidimensionalRange(std::move(other));
}

void IndexSet::insert(MultidimensionalRange other) {
  assert((!isInitialized() || other.rank() == rank()) && "Incompatible rank");

  // Filter out the already existing points.
  llvm::SmallVector<MultidimensionalRange> nonOverlappingRanges;

  if (!getRoot()) {
    nonOverlappingRanges.push_back(other);
  } else {
    llvm::SmallVector<MultidimensionalRange> current;
    current.push_back(other);

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
            assert(other.contains(diff));
            nonOverlappingRanges.push_back(std::move(diff));
          }
        }
      }

      current = std::move(next);
    }
  }

  // Add the remaining points.
  for (MultidimensionalRange &range : nonOverlappingRanges) {
    RTreeCRTP::insert(std::move(range));
  }

  assert(llvm::all_of(other,
                      [&](const Point &point) { return contains(point); }) &&
         "Not all points have been inserted");
}

IndexSet &IndexSet::operator+=(MultidimensionalRange other) {
  insert(std::move(other));
  return *this;
}

IndexSet &IndexSet::operator+=(const IndexSet &other) {
  RTreeCRTP::insert(other);
  return *this;
}

IndexSet IndexSet::operator+(Point other) const {
  IndexSet result(*this);
  result += std::move(other);
  return result;
}

IndexSet IndexSet::operator+(MultidimensionalRange other) const {
  IndexSet result(*this);
  result += std::move(other);
  return result;
}

IndexSet IndexSet::operator+(const IndexSet &other) const {
  IndexSet result(*this);
  result += other;
  return result;
}

IndexSet &IndexSet::operator-=(const Point &other) {
  return *this -= MultidimensionalRange(other);
}

void IndexSet::remove(const MultidimensionalRange &other) {
  assert((!isInitialized() || other.rank() == rank()) && "Incompatible rank");

  if (empty()) {
    return;
  }

  llvm::SmallVector<Node *> overlappingNodes;

  walkOverlappingLeafNodes(
      other, [&](Node &node) { overlappingNodes.push_back(&node); });

  for (Node *node : overlappingNodes) {
    assert(node->isLeaf());
    bool modified = false;

    llvm::SmallVector<Object> differences;

    for (Object &value : node->objects) {
      if (value->overlaps(other)) {
        modified = true;
        llvm::append_range(differences, value->subtract(other));
      } else {
        differences.emplace_back(std::move(value));
      }
    }

    node->objects = std::move(differences);

    if (modified) {
      // Split and collapse nodes, if necessary.
      adjustTree(node);
      break;
    }
  }

  assert(isValid());

  assert(llvm::all_of(other,
                      [&](const Point &point) { return !contains(point); }) &&
         "Not all points have been removed");
}

IndexSet &IndexSet::operator-=(const MultidimensionalRange &other) {
  remove(other);

  assert(llvm::all_of(other,
                      [&](const Point &point) { return !contains(point); }) &&
         "Not all points have been removed");

  return *this;
}

IndexSet &IndexSet::operator-=(const IndexSet &other) {
  RTreeCRTP::remove(other);

  assert(llvm::all_of(other,
                      [&](const Point &point) { return !contains(point); }) &&
         "Not all points have been removed");

  return *this;
}

IndexSet IndexSet::operator-(const Point &other) const {
  IndexSet result(*this);
  result -= other;
  return result;
}

IndexSet IndexSet::operator-(const MultidimensionalRange &other) const {
  IndexSet result(*this);
  result -= other;
  return result;
}

IndexSet IndexSet::operator-(const IndexSet &other) const {
  IndexSet result(*this);
  result -= other;
  return result;
}

size_t IndexSet::flatSize() const {
  size_t result = 0;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    result += range.flatSize();
  }

  return result;
}

IndexSet::const_point_iterator IndexSet::begin() const {
  return IndexSet::PointIterator::begin(*this);
}

IndexSet::const_point_iterator IndexSet::end() const {
  return IndexSet::PointIterator::end(*this);
}

IndexSet::const_range_iterator IndexSet::rangesBegin() const {
  return IndexSet::RangeIterator::begin(*this);
}

IndexSet::const_range_iterator IndexSet::rangesEnd() const {
  return IndexSet::RangeIterator::end(*this);
}

bool IndexSet::contains(const Point &other) const {
  return contains(MultidimensionalRange(other));
}

bool IndexSet::contains(const MultidimensionalRange &other) const {
  if (empty()) {
    return false;
  }

  std::queue<MultidimensionalRange> remainingRanges;
  remainingRanges.push(other);
  bool changesDetected = true;

  while (!remainingRanges.empty() && changesDetected) {
    changesDetected = false;
    const MultidimensionalRange &range = remainingRanges.front();

    llvm::SmallVector<const Node *> nodes;

    if (getRoot()->getBoundary().overlaps(range)) {
      nodes.push_back(getRoot());
    }

    while (!nodes.empty() && !changesDetected) {
      const Node *node = nodes.pop_back_val();

      for (const auto &child : node->children) {
        if (child->getBoundary().overlaps(range)) {
          nodes.push_back(child.get());
        }
      }

      for (const Object &object : node->objects) {
        if (range.overlaps(*object)) {
          for (MultidimensionalRange &difference : range.subtract(*object)) {
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

bool IndexSet::contains(const IndexSet &other) const {
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
        for (const Object &object : rhs->objects) {
          llvm::SmallVector<MultidimensionalRange, 3> remainingRanges;
          remainingRanges.push_back(*object);

          for (const auto &lhs : lhsNodes) {
            assert(lhs->isLeaf());

            for (const Object &lhsRange : lhs->objects) {
              llvm::SmallVector<MultidimensionalRange, 3> newRemaining;

              for (const auto &remainingRange : remainingRanges) {
                for (auto &diff : remainingRange.subtract(*lhsRange)) {
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

bool IndexSet::overlaps(const MultidimensionalRange &other) const {
  if (empty()) {
    return false;
  }

  llvm::SmallVector<const Node *> nodes;

  if (getRoot()->getBoundary().overlaps(other)) {
    nodes.push_back(getRoot());
  }

  while (!nodes.empty()) {
    auto node = nodes.pop_back_val();

    for (const auto &child : node->children) {
      if (child->getBoundary().overlaps(other)) {
        nodes.push_back(child.get());
      }
    }

    for (const Object &object : node->objects) {
      if (object->overlaps(other)) {
        return true;
      }
    }
  }

  return false;
}

bool IndexSet::overlaps(const IndexSet &other) const {
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
        for (const Object &lhsRange : lhs->objects) {
          for (const Object &rhsRange : rhs->objects) {
            if (lhsRange->overlaps(*rhsRange)) {
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

IndexSet IndexSet::intersect(const MultidimensionalRange &other) const {
  if (empty()) {
    return *this;
  }

  llvm::SmallVector<const Node *> nodes;

  if (getRoot()->getBoundary().overlaps(other)) {
    nodes.push_back(getRoot());
  } else {
    return emptyCopy();
  }

  llvm::SmallVector<MultidimensionalRange> resultRanges;

  while (!nodes.empty()) {
    const Node *node = nodes.pop_back_val();

    for (const auto &child : node->children) {
      if (child->getBoundary().overlaps(other)) {
        nodes.push_back(child.get());
      }
    }

    for (const auto &object : node->objects) {
      if (object.getBoundary().overlaps(other)) {
        resultRanges.push_back(other.intersect(*object));
      }
    }
  }

  return {std::move(resultRanges)};
}

IndexSet IndexSet::intersect(const IndexSet &other) const {
  if (empty()) {
    return *this;
  }

  if (other.empty()) {
    return emptyCopy();
  }

  using OverlappingNodePair = std::pair<const Node *, const Node *>;
  llvm::SmallVector<OverlappingNodePair> overlappingNodes;

  const Node *lhsRoot = getRoot();
  const Node *rhsRoot = other.getRoot();

  if (lhsRoot->getBoundary().overlaps(rhsRoot->getBoundary())) {
    overlappingNodes.emplace_back(lhsRoot, rhsRoot);
  }

  llvm::SmallVector<MultidimensionalRange> resultRanges;

  while (!overlappingNodes.empty()) {
    OverlappingNodePair overlappingNodePair = overlappingNodes.pop_back_val();

    const Node *lhs = overlappingNodePair.first;
    const Node *rhs = overlappingNodePair.second;

    if (lhs->isLeaf()) {
      if (rhs->isLeaf()) {
        for (const Object &lhsRange : lhs->objects) {
          for (const Object &rhsRange : rhs->objects) {
            if (lhsRange->overlaps(*rhsRange)) {
              resultRanges.push_back(lhsRange->intersect(*rhsRange));
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

  return {std::move(resultRanges)};
}

IndexSet IndexSet::complement(const MultidimensionalRange &other) const {
  if (empty()) {
    return {other};
  }

  llvm::SmallVector<MultidimensionalRange> resultRanges;

  llvm::SmallVector<MultidimensionalRange> currentRanges;
  currentRanges.push_back(other);

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    llvm::SmallVector<MultidimensionalRange> nextRanges;

    for (const MultidimensionalRange &curr : currentRanges) {
      for (MultidimensionalRange &diff : curr.subtract(range)) {
        if (overlaps(diff)) {
          nextRanges.push_back(std::move(diff));
        } else {
          resultRanges.push_back(std::move(diff));
        }
      }
    }

    currentRanges = std::move(nextRanges);
  }

  return {std::move(resultRanges)};
}

IndexSet IndexSet::slice(const llvm::BitVector &filter) const {
  if (empty()) {
    return *this;
  }

  llvm::SmallVector<MultidimensionalRange> resultRanges;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    resultRanges.push_back(range.slice(filter));
  }

  return {std::move(resultRanges)};
}

IndexSet IndexSet::takeFirstDimensions(size_t n) const {
  assert(!isInitialized() || n <= rank());

  if (empty()) {
    return *this;
  }

  llvm::SmallVector<MultidimensionalRange> resultRanges;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    resultRanges.push_back(range.takeFirstDimensions(n));
  }

  return {std::move(resultRanges)};
}

IndexSet IndexSet::takeLastDimensions(size_t n) const {
  assert(!isInitialized() || n <= rank());

  if (empty()) {
    return *this;
  }

  llvm::SmallVector<MultidimensionalRange> resultRanges;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    resultRanges.push_back(range.takeLastDimensions(n));
  }

  return {std::move(resultRanges)};
}

IndexSet
IndexSet::takeDimensions(const llvm::SmallBitVector &dimensions) const {
  assert(!isInitialized() || dimensions.size() == rank());

  if (empty()) {
    return *this;
  }

  llvm::SmallVector<MultidimensionalRange> resultRanges;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    resultRanges.push_back(range.takeDimensions(dimensions));
  }

  return {std::move(resultRanges)};
}

IndexSet IndexSet::dropFirstDimensions(size_t n) const {
  assert(!isInitialized() || n < rank());

  if (empty()) {
    return *this;
  }

  llvm::SmallVector<MultidimensionalRange> resultRanges;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    resultRanges.push_back(range.dropFirstDimensions(n));
  }

  return {std::move(resultRanges)};
}

IndexSet IndexSet::dropLastDimensions(size_t n) const {
  assert(!isInitialized() || n < rank());

  if (empty()) {
    return *this;
  }

  llvm::SmallVector<MultidimensionalRange> resultRanges;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    resultRanges.push_back(range.dropLastDimensions(n));
  }

  return {std::move(resultRanges)};
}

IndexSet IndexSet::prepend(const IndexSet &other) const {
  if (empty()) {
    return *this;
  }

  llvm::SmallVector<MultidimensionalRange> resultRanges;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    for (const MultidimensionalRange &otherRange :
         llvm::make_range(other.rangesBegin(), other.rangesEnd())) {
      resultRanges.push_back(range.prepend(otherRange));
    }
  }

  return {std::move(resultRanges)};
}

IndexSet IndexSet::append(const IndexSet &other) const {
  if (empty()) {
    return *this;
  }

  llvm::SmallVector<MultidimensionalRange> resultRanges;

  for (const MultidimensionalRange &range :
       llvm::make_range(rangesBegin(), rangesEnd())) {
    for (const MultidimensionalRange &otherRange :
         llvm::make_range(other.rangesBegin(), other.rangesEnd())) {
      resultRanges.push_back(range.append(otherRange));
    }
  }

  return {std::move(resultRanges)};
}

void IndexSet::postObjectInsertionHook(Node &node) {
  llvm::sort(node.objects, [](const Object &first, const Object &second) {
    return first.getBoundary() < second.getBoundary();
  });

  merge(node.objects);
}

void IndexSet::postObjectRemovalHook(Node &node) {
  llvm::sort(node.objects, [](const Object &first, const Object &second) {
    return first.getBoundary() < second.getBoundary();
  });

  merge(node.objects);
}
} // namespace marco::modeling

namespace {
bool shouldSplitRange(const MultidimensionalRange &range,
                      const MultidimensionalRange &grid, size_t dimension) {
  assert(range.rank() == grid.rank());
  assert(dimension < range.rank());

  return grid[dimension].overlaps(range[dimension]);
}

std::list<MultidimensionalRange> splitRange(const MultidimensionalRange &range,
                                            const MultidimensionalRange &grid,
                                            size_t dimension) {
  assert(range[dimension].overlaps(grid[dimension]));
  std::list<MultidimensionalRange> result;

  llvm::SmallVector<Range> newDimensionRanges;
  newDimensionRanges.push_back(range[dimension].intersect(grid[dimension]));

  for (Range &diff : range[dimension].subtract(grid[dimension])) {
    newDimensionRanges.push_back(std::move(diff));
  }

  llvm::SmallVector<Range> newRanges;

  for (Range &newDimensionRange : newDimensionRanges) {
    newRanges.clear();

    for (size_t i = 0; i < range.rank(); ++i) {
      if (i == dimension) {
        newRanges.push_back(std::move(newDimensionRange));
      } else {
        newRanges.push_back(range[i]);
      }
    }

    result.push_back(MultidimensionalRange(newRanges));
  }

  return result;
}

void splitRangesToMinGrid(std::list<MultidimensionalRange> &ranges) {
  if (ranges.empty()) {
    return;
  }

  size_t rank = ranges.front().rank();

  for (size_t dimension = 0; dimension < rank - 1; ++dimension) {
    for (auto it1 = ranges.begin(); it1 != ranges.end(); ++it1) {
      auto it2 = ranges.begin();

      while (it2 != ranges.end()) {
        if (it1 == it2) {
          ++it2;
          continue;
        }

        const MultidimensionalRange &range = *it2;
        const MultidimensionalRange &grid = *it1;

        if (shouldSplitRange(range, grid, dimension)) {
          std::list<MultidimensionalRange> splitRanges =
              splitRange(range, grid, dimension);

          ranges.splice(it2, splitRanges);
          it2 = ranges.erase(it2);
        } else {
          ++it2;
        }
      }
    }
  }
}

void removeDuplicateRanges(std::list<MultidimensionalRange> &ranges) {
  assert(llvm::is_sorted(ranges, [](const MultidimensionalRange &first,
                                    const MultidimensionalRange &second) {
    return first < second;
  }));

  auto it = ranges.begin();

  if (it == ranges.end()) {
    return;
  }

  while (it != ranges.end()) {
    auto next = std::next(it);

    while (next != ranges.end() && *next == *it) {
      ranges.erase(next);
      next = std::next(it);
    }

    ++it;
  }
}

void mergeRanges(std::list<MultidimensionalRange> &ranges) {
  using It = std::list<MultidimensionalRange>::iterator;

  auto findCandidates = [&](It begin, It end) -> std::tuple<It, It, size_t> {
    for (auto it1 = begin; it1 != end; ++it1) {
      for (auto it2 = std::next(it1); it2 != end; ++it2) {
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
}
} // namespace

namespace marco::modeling {
void IndexSet::getCompactRanges(
    llvm::SmallVectorImpl<MultidimensionalRange> &result) const {
  std::list ranges(rangesBegin(), rangesEnd());

  splitRangesToMinGrid(ranges);
  ranges.sort();
  removeDuplicateRanges(ranges);
  mergeRanges(ranges);

  llvm::append_range(result, llvm::make_range(ranges.begin(), ranges.end()));
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const IndexSet &indexSet) {
  os << "{";

  bool separator = false;

  for (const MultidimensionalRange &range :
       llvm::make_range(indexSet.rangesBegin(), indexSet.rangesEnd())) {
    if (separator) {
      os << ", ";
    }

    separator = true;
    os << range;
  }

  return os << "}";
}
} // namespace marco::modeling

//===---------------------------------------------------------------------===//
// Range iterator
//===---------------------------------------------------------------------===//

namespace marco::modeling {
IndexSet::RangeIterator::RangeIterator(base_iterator baseIterator)
    : baseIterator(std::move(baseIterator)) {}

IndexSet::RangeIterator IndexSet::RangeIterator::begin(const IndexSet &obj) {
  return {obj.objectsBegin()};
}

IndexSet::RangeIterator IndexSet::RangeIterator::end(const IndexSet &obj) {
  return {obj.objectsEnd()};
}

bool IndexSet::RangeIterator::operator==(
    const IndexSet::RangeIterator &other) const {
  return baseIterator == other.baseIterator;
}

bool IndexSet::RangeIterator::operator!=(
    const IndexSet::RangeIterator &other) const {
  return baseIterator != other.baseIterator;
}

IndexSet::RangeIterator &IndexSet::RangeIterator::operator++() {
  ++baseIterator;
  return *this;
}

IndexSet::RangeIterator IndexSet::RangeIterator::operator++(int) {
  IndexSet::RangeIterator result(*this);
  ++(*this);
  return result;
}

IndexSet::RangeIterator::reference IndexSet::RangeIterator::operator*() const {
  return *baseIterator;
}
} // namespace marco::modeling

//===---------------------------------------------------------------------===//
// Point iterator
//===---------------------------------------------------------------------===//

namespace marco::modeling {
IndexSet::PointIterator::PointIterator(
    IndexSet::RangeIterator currentRangeIt, IndexSet::RangeIterator endRangeIt,
    std::optional<MultidimensionalRange::const_iterator> currentPointIt,
    std::optional<MultidimensionalRange::const_iterator> endPointIt)
    : currentRangeIt(std::move(currentRangeIt)),
      endRangeIt(std::move(endRangeIt)),
      currentPointIt(std::move(currentPointIt)),
      endPointIt(std::move(endPointIt)) {
  fetchNext();
}

IndexSet::PointIterator
IndexSet::PointIterator::begin(const IndexSet &indexSet) {
  auto currentRangeIt = indexSet.rangesBegin();
  auto endRangeIt = indexSet.rangesEnd();

  if (currentRangeIt == endRangeIt) {
    // There are no ranges. The current range iterator is already
    // past-the-end, and thus we must avoid dereferencing it.

    PointIterator it(currentRangeIt, endRangeIt, std::nullopt, std::nullopt);
    return {std::move(it)};
  }

  auto currentPointIt = (*currentRangeIt).begin();
  auto endPointIt = (*currentRangeIt).end();
  PointIterator it(currentRangeIt, endRangeIt, currentPointIt, endPointIt);

  return {std::move(it)};
}

IndexSet::PointIterator IndexSet::PointIterator::end(const IndexSet &indexSet) {
  PointIterator it(indexSet.rangesEnd(), indexSet.rangesEnd(), std::nullopt,
                   std::nullopt);

  return {std::move(it)};
}

bool IndexSet::PointIterator::operator==(
    const IndexSet::PointIterator &other) const {
  return currentRangeIt == other.currentRangeIt &&
         currentPointIt == other.currentPointIt;
}

bool IndexSet::PointIterator::operator!=(
    const IndexSet::PointIterator &other) const {
  return currentRangeIt != other.currentRangeIt ||
         currentPointIt != other.currentPointIt;
}

IndexSet::PointIterator &IndexSet::PointIterator::operator++() {
  advance();
  return *this;
}

IndexSet::PointIterator IndexSet::PointIterator::operator++(int) {
  IndexSet::PointIterator result(*this);
  ++(*this);
  return result;
}

IndexSet::PointIterator::value_type IndexSet::PointIterator::operator*() const {
  return **currentPointIt;
}

bool IndexSet::PointIterator::shouldProceed() const {
  if (currentRangeIt == endRangeIt) {
    return false;
  }

  return currentPointIt == endPointIt;
}

void IndexSet::PointIterator::fetchNext() {
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

void IndexSet::PointIterator::advance() {
  ++(*currentPointIt);
  fetchNext();
}
} // namespace marco::modeling

//===----------------------------------------------------------------------===//
// DenseMapInfo
//===----------------------------------------------------------------------===//

namespace llvm {
IndexSet llvm::DenseMapInfo<IndexSet>::getEmptyKey() {
  IndexSet result;
  result += llvm::DenseMapInfo<MultidimensionalRange>::getEmptyKey();
  return result;
}

IndexSet llvm::DenseMapInfo<IndexSet>::getTombstoneKey() {
  IndexSet result;
  result += llvm::DenseMapInfo<MultidimensionalRange>::getTombstoneKey();
  return result;
}

unsigned DenseMapInfo<IndexSet>::getHashValue(const IndexSet &val) {
  return hash_value(val);
}

bool DenseMapInfo<IndexSet>::isEqual(const IndexSet &lhs, const IndexSet &rhs) {
  return lhs == rhs;
}
} // namespace llvm
