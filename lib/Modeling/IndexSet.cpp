#include "marco/Modeling/IndexSet.h"
#include "llvm/ADT/SmallVector.h"
#include <stack>
#include <queue>

using namespace ::marco;
using namespace ::marco::modeling;

namespace
{
  /// A node of the R-Tree
  struct Node
  {
    Node(Node* parent, MultidimensionalRange boundary);

    Node(const Node& other);

    ~Node();

    Node& operator=(Node&& other);

    bool isRoot() const;

    bool isLeaf() const;

    const MultidimensionalRange& getBoundary() const;

    /// Recalculate the MBR containing all the values / children.
    void recalcBoundary();

    /// Get the multidimensional space (area, volume, etc.) covered by the MBR.
    unsigned int getFlatSize() const;

    /// Get the amount of children or contained values.
    size_t fanOut() const;

    /// Add a child node.
    void add(std::unique_ptr<Node> child);

    /// Add a value.
    void add(MultidimensionalRange value);

    public:
      Node* parent;

    private:
      /// MBR containing all the children / values.
      MultidimensionalRange boundary;

      // The flat size of the MBR is stored internally so that it doesn't
      // need to be computed all the times (which has an O(d) cost, with d
      // equal to the rank).
      unsigned int flatSize;

    public:
      std::vector<std::unique_ptr<Node>> children;
      std::vector<MultidimensionalRange> values;
  };
}

namespace
{
  template<typename T>
  const MultidimensionalRange& getShape(const T& obj);

  template<>
  const MultidimensionalRange& getShape<MultidimensionalRange>(const MultidimensionalRange& range)
  {
    return range;
  }

  template<>
  const MultidimensionalRange& getShape<std::unique_ptr<Node>>(const std::unique_ptr<Node>& node)
  {
    return node->getBoundary();
  }

  /// Get the minimum bounding (multidimensional) range.
  MultidimensionalRange getMBR(const MultidimensionalRange& first, const MultidimensionalRange& second)
  {
    assert(first.rank() == second.rank() && "Can't compute the MBR between ranges on two different hyper-spaces");
    std::vector<Range> ranges;

    for (size_t i = 0; i < first.rank(); ++i) {
      ranges.emplace_back(
          std::min(first[i].getBegin(), second[i].getBegin()),
          std::max(first[i].getEnd(), second[i].getEnd()));
    }

    return MultidimensionalRange(std::move(ranges));
  }

  template<typename T>
  MultidimensionalRange getMBR(llvm::ArrayRef<T> objects)
  {
    assert(!objects.empty());
    MultidimensionalRange result = getShape(objects[0]);

    for (size_t i = 1; i < objects.size(); ++i) {
      result = getMBR(result, getShape(objects[i]));
    }

    return result;
  }

  Node::Node(Node* parent, MultidimensionalRange boundary)
      : parent(parent),
        boundary(boundary),
        flatSize(boundary.flatSize())
  {
    assert(this != parent && "A node can't be the parent of itself");
  }

  Node::Node(const Node& other)
    : parent(other.parent),
      boundary(other.boundary),
      flatSize(other.flatSize)
  {
    for (const auto& child : other.children) {
      auto& newChild = children.emplace_back(std::make_unique<Node>(*child));
      newChild->parent = this;
    }

    for (const auto& value : other.values) {
      values.push_back(value);
    }
  }

  Node::~Node() = default;

  Node& Node::operator=(Node&& other) = default;

  bool Node::isRoot() const
  {
    return parent == nullptr;
  }

  bool Node::isLeaf() const
  {
    auto result = children.empty();
    assert(result || values.empty());
    return result;
  }

  const MultidimensionalRange& Node::getBoundary() const
  {
    return boundary;
  }

  void Node::recalcBoundary()
  {
    if (isLeaf()) {
      boundary = getMBR(llvm::makeArrayRef(values));
    } else {
      boundary = getMBR(llvm::makeArrayRef(children));
    }

    // Check the validity of the MBR
    assert(llvm::all_of(children, [&](const auto& child) {
      return boundary.contains(getShape(child));
    }));

    assert(llvm::all_of(values, [&](const auto& value) {
      return boundary.contains(getShape(value));
    }));

    // Update the covered area
    flatSize = boundary.flatSize();
  }

  unsigned int Node::getFlatSize() const
  {
    return flatSize;
  }

  size_t Node::fanOut() const
  {
    if (isLeaf()) {
      return values.size();
    }

    return children.size();
  }

  void Node::add(std::unique_ptr<Node> child)
  {
    assert(values.empty());
    children.push_back(std::move(child));
  }

  void Node::add(MultidimensionalRange value)
  {
    assert(children.empty());
    values.push_back(std::move(value));
  }

  /// Select two entries to be the first elements of the new groups.
  template<typename T>
  std::pair<size_t, size_t> pickSeeds(llvm::ArrayRef<T> entries)
  {
    std::vector<std::tuple<size_t, size_t, size_t>> candidates;

    for (size_t i = 0; i < entries.size(); ++i) {
      for (size_t j = i + 1; j < entries.size(); ++j) {
        auto mbr = getMBR(getShape(entries[i]), getShape(entries[j]));
        auto difference = mbr.flatSize() - getShape(entries[i]).flatSize() - getShape(entries[j]).flatSize();
        candidates.emplace_back(i, j, difference);
      }
    }

    auto it = std::max_element(candidates.begin(), candidates.end(), [](const auto& first, const auto& second) {
      return std::get<2>(first) < std::get<2>(second);
    });

    return std::make_pair(std::get<0>(*it), std::get<1>(*it));
  }

  /// Select one remaining entry for classification in a group.
  template<typename T>
  size_t pickNext(
      llvm::ArrayRef<T> entries,
      const std::set<size_t>& alreadyProcessedEntries,
      const Node& firstNode,
      const Node& secondNode)
  {
    assert(!entries.empty());

    // Determine the cost of putting each entry in each group
    std::vector<std::pair<size_t, size_t>> costs;

    for (const auto& entry : llvm::enumerate(entries)) {
      if (alreadyProcessedEntries.find(entry.index()) != alreadyProcessedEntries.end()) {
        continue;
      }

      auto d1 = getMBR(getShape(entry.value()), firstNode.getBoundary()).flatSize() - firstNode.getBoundary().flatSize();
      auto d2 = getMBR(getShape(entry.value()), secondNode.getBoundary()).flatSize() - secondNode.getBoundary().flatSize();
      auto minMax = std::minmax(d1, d2);
      costs.emplace_back(entry.index(), minMax.second - minMax.first);
    }

    // Find the entry with the greatest preference for one group
    assert(!costs.empty());

    auto it = std::max_element(
        costs.begin(), costs.end(),
        [](const auto& first, const auto& second) {
          return first.second < second.second;
        });

    return it->first;
  }
}

namespace marco::modeling
{
  /// R-Tree IndexSet implementation
  class IndexSet::Impl
  {
    public:
      class Iterator;
      using const_iterator = Iterator;

      Impl(size_t minElements = 4, size_t maxElements = 16);

      Impl(llvm::ArrayRef<Point> points);

      Impl(llvm::ArrayRef<MultidimensionalRange> ranges);

      Impl(const Impl& other);

      Impl(Impl&& other);

      ~Impl();

      Impl& operator=(const Impl& other);

      friend void swap(Impl& first, Impl& second);

      friend std::ostream& operator<<(std::ostream& os, const IndexSet::Impl& obj);

      bool operator==(const Point& rhs) const;

      bool operator==(const MultidimensionalRange& rhs) const;

      bool operator==(const IndexSet::Impl& rhs) const;

      bool operator!=(const Point& rhs) const;

      bool operator!=(const MultidimensionalRange& rhs) const;

      bool operator!=(const IndexSet::Impl& rhs) const;

      IndexSet::Impl& operator+=(const Point& rhs);

      IndexSet::Impl& operator+=(const MultidimensionalRange& rhs);

      IndexSet::Impl& operator+=(const IndexSet::Impl& rhs);

      IndexSet::Impl operator+(const Point& rhs) const;

      IndexSet::Impl operator+(const MultidimensionalRange& rhs) const;

      IndexSet::Impl operator+(const IndexSet::Impl& rhs) const;

      IndexSet::Impl& operator-=(const MultidimensionalRange& rhs);

      IndexSet::Impl& operator-=(const IndexSet::Impl& rhs);

      IndexSet::Impl operator-(const MultidimensionalRange& rhs) const;

      IndexSet::Impl operator-(const IndexSet::Impl& rhs) const;

      bool empty() const;

      size_t rank() const;

      size_t flatSize() const;

      void clear();

      const_iterator begin() const;

      const_iterator end() const;

      bool contains(const Point& other) const;

      bool contains(const MultidimensionalRange& other) const;

      bool contains(const IndexSet::Impl& other) const;

      bool overlaps(const MultidimensionalRange& other) const;

      bool overlaps(const IndexSet::Impl& other) const;

      IndexSet::Impl intersect(const MultidimensionalRange& other) const;

      IndexSet::Impl intersect(const IndexSet::Impl& other) const;

      IndexSet::Impl complement(const MultidimensionalRange& other) const;

      const Node* getRoot() const;

    private:
      /// Select a leaf node in which to place a new entry.
      Node* chooseLeaf(const MultidimensionalRange& entry) const;

      template<typename T>
      std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>> splitNode(
          Node& node,
          std::function<std::vector<T>&(Node&)> containerFn)
      {
        auto seeds = pickSeeds(llvm::makeArrayRef(containerFn(node)));

        std::set<size_t> movedValues;

        auto moveValueFn = [&](Node& destination, size_t valueIndex) {
          if (movedValues.find(valueIndex) == movedValues.end()) {
            containerFn(destination).push_back(std::move(containerFn(node)[valueIndex]));
            movedValues.insert(valueIndex);
          }
        };

        auto firstNew = std::make_unique<Node>(node.parent, getShape(containerFn(node)[seeds.first]));
        moveValueFn(*firstNew, seeds.first);

        auto secondNew = std::make_unique<Node>(node.parent, getShape(containerFn(node)[seeds.second]));
        moveValueFn(*secondNew, seeds.second);

        // Keep processing the values until all of them have been assigned to the new nodes
        while (movedValues.size() != containerFn(node).size()) {
          auto remaining = containerFn(node).size() - movedValues.size();

          // If one group has so few entries that all the rest must be assigned to it in
          // order for it to have the minimum number of element, then assign them and stop.

          if (containerFn(*firstNew).size() + remaining == minElements) {
            for (auto& value : llvm::enumerate(containerFn(node))) {
              moveValueFn(*firstNew, value.index());
            }

            assert(movedValues.size() == containerFn(node).size());
            break;

          } else if (containerFn(*secondNew).size() + remaining == minElements) {
            for (auto& value : llvm::enumerate(containerFn(node))) {
              moveValueFn(*secondNew, value.index());
            }

            assert(movedValues.size() == containerFn(node).size());
            break;
          }

          // Choose the next entry to assign.
          auto next = pickNext(llvm::makeArrayRef(containerFn(node)), movedValues, *firstNew, *secondNew);
          assert(movedValues.find(next) == movedValues.end());

          // Add it to the group whose covering rectangle will have to be
          // enlarged least to accommodate it. Resolve ties by adding the entry to
          // the group with smaller area, then to the one with fewer entries, then
          // to either of them.

          auto firstEnlargement = getMBR(getShape(containerFn(node)[next]), firstNew->getBoundary()).flatSize() - firstNew->getFlatSize();
          auto secondEnlargement = getMBR(getShape(containerFn(node)[next]), secondNew->getBoundary()).flatSize() - secondNew->getFlatSize();

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
        assert(containerFn(*firstNew).size() <= maxElements);

        assert(containerFn(*secondNew).size() >= minElements);
        assert(containerFn(*secondNew).size() <= maxElements);

        // Update the boundaries of the nodes
        firstNew->recalcBoundary();
        secondNew->recalcBoundary();

        return std::make_pair(std::move(firstNew), std::move(secondNew));
      }

      std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>> splitNode(Node& node);

      /// Check if all the invariants are respected.
      bool isValid() const;

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

  class IndexSet::Impl::Iterator
  {
    public:
      using iterator_category = std::input_iterator_tag;
      using value_type = MultidimensionalRange;
      using difference_type = std::ptrdiff_t;
      using pointer = const MultidimensionalRange*;
      using reference = const MultidimensionalRange&;

      static Iterator begin(const IndexSet::Impl& indexSet);

      static Iterator end(const IndexSet::Impl& indexSet);

      bool operator==(const Iterator& it) const;

      bool operator!=(const Iterator& it) const;

      Iterator& operator++();

      Iterator operator++(int);

      reference operator*() const;

    private:
      Iterator(const Node* root);

      void fetchNextLeaf();

    private:
      std::stack<const Node*> nodes;
      const Node* node;
      size_t valueIndex;
  };
}

static void merge(std::vector<MultidimensionalRange>& ranges)
{
  if (ranges.empty()) {
    return;
  }

  using It = std::vector<MultidimensionalRange>::iterator;

  auto findCandidates = [&](It begin, It end) -> std::tuple<It, It, size_t> {
    for (It it1 = begin; it1 != end; ++it1) {
      for (It it2 = std::next(it1); it2 != end; ++it2) {
        if (auto mergePossibility = it1->canBeMerged(*it2); mergePossibility.first) {
          return std::make_tuple(it1, it2, mergePossibility.second);
        }
      }
    }

    return std::make_tuple(end, end, 0);
  };

  auto candidates = findCandidates(ranges.begin(), ranges.end());

  while (std::get<0>(candidates) != ranges.end() && std::get<1>(candidates) != ranges.end()) {
    auto& first = std::get<0>(candidates);
    auto& second = std::get<1>(candidates);
    size_t dimension = std::get<2>(candidates);

    *first = first->merge(*second, dimension);
    ranges.erase(second);
    candidates = findCandidates(ranges.begin(), ranges.end());
  }

  assert(!ranges.empty());
}

/// Check that all the children of a node has the correct parent set.
static bool checkParentRelationships(const IndexSet::Impl& indexSet)
{
  std::stack<const Node*> nodes;

  if (auto root = indexSet.getRoot(); root != nullptr) {
    nodes.push(root);
  }

  while (!nodes.empty()) {
    auto node = nodes.top();
    nodes.pop();

    for (const auto& child : node->children) {
      if (child->parent != node) {
        return false;
      }

      nodes.push(child.get());
    }
  }

  return true;
}

/// Check the correctness of the MBR of all the nodes.
static bool checkMBRsInvariant(const IndexSet::Impl& indexSet)
{
  std::stack<const Node*> nodes;

  if (auto root = indexSet.getRoot(); root != nullptr) {
    nodes.push(root);
  }

  while (!nodes.empty()) {
    auto node = nodes.top();
    nodes.pop();

    if (node->isLeaf()) {
      if (node->getBoundary() != getMBR(llvm::makeArrayRef(node->values))) {
        return false;
      }
    } else {
      if (node->getBoundary() != getMBR(llvm::makeArrayRef(node->children))) {
        return false;
      }
    }

    for (const auto& child : node->children) {
      nodes.push(child.get());
    }
  }

  return true;
}

/// Check that all the nodes have between a minimum and a maximum
/// amount of out edges (apart the root node).
static bool checkFanOutInvariant(const IndexSet::Impl& indexSet)
{
  std::stack<const Node*> nodes;

  if (auto root = indexSet.getRoot(); root != nullptr) {
    for (const auto& child : root->children) {
      nodes.push(child.get());
    }
  }

  while (!nodes.empty()) {
    auto node = nodes.top();
    nodes.pop();

    if (auto size = node->fanOut(); size < indexSet.minElements || size > indexSet.maxElements) {
      return false;
    }

    for (const auto& child : node->children) {
      nodes.push(child.get());
    }
  }

  return true;
}

namespace marco::modeling
{
  IndexSet::Impl::Impl(size_t minElements, size_t maxElements)
    : minElements(minElements),
      maxElements(maxElements),
      root(nullptr),
      initialized(false),
      allowedRank(0)
  {
    assert(maxElements > 0);
    assert(minElements <= maxElements / 2);
  }

  IndexSet::Impl::Impl(llvm::ArrayRef<Point> points) : Impl()
  {
    for (const auto& point: points) {
      *this += point;
    }
  }

  IndexSet::Impl::Impl(llvm::ArrayRef<MultidimensionalRange> ranges) : Impl()
  {
    for (const auto& range: ranges) {
      *this += range;
    }
  }

  IndexSet::Impl::Impl(const Impl& other)
    : minElements(other.minElements),
      maxElements(other.maxElements),
      root(other.root == nullptr ? nullptr : std::make_unique<Node>(*other.root)),
      initialized(other.initialized),
      allowedRank(other.allowedRank)
  {
  }

  IndexSet::Impl::Impl(Impl&& other) = default;

  IndexSet::Impl::~Impl() = default;

  IndexSet::Impl& IndexSet::Impl::operator=(const IndexSet::Impl& other)
  {
    IndexSet::Impl result(other);
    swap(*this, result);
    return *this;
  }

  void swap(IndexSet::Impl& first, IndexSet::Impl& second)
  {
    using std::swap;
    swap(const_cast<size_t&>(first.minElements), const_cast<size_t&>(second.minElements));
    swap(const_cast<size_t&>(first.maxElements), const_cast<size_t&>(second.maxElements));
    swap(first.root, second.root);
    swap(first.initialized, second.initialized);
    swap(first.allowedRank, second.allowedRank);
  }

  std::ostream& operator<<(std::ostream& os, const IndexSet::Impl& obj)
  {
    os << "{";

    bool separator = false;

    for (const auto& range : obj) {
      if (separator) {
        os << ", ";
      }

      separator = true;
      os << range;
    }

    os << "}";
    return os;
  }

  bool IndexSet::Impl::operator==(const Point& rhs) const
  {
    return *this == IndexSet::Impl(rhs);
  }

  bool IndexSet::Impl::operator==(const MultidimensionalRange& rhs) const
  {
    return *this == IndexSet::Impl(rhs);
  }

  bool IndexSet::Impl::operator==(const IndexSet::Impl& rhs) const
  {
    return contains(rhs) && rhs.contains(*this);
  }

  bool IndexSet::Impl::operator!=(const Point& rhs) const
  {
    return *this != IndexSet::Impl(rhs);
  }

  bool IndexSet::Impl::operator!=(const MultidimensionalRange& rhs) const
  {
    return *this != IndexSet::Impl(rhs);
  }

  bool IndexSet::Impl::operator!=(const IndexSet::Impl& rhs) const
  {
    return !contains(rhs) || !rhs.contains(*this);
  }

  IndexSet::Impl& IndexSet::Impl::operator+=(const Point& rhs)
  {
    llvm::SmallVector<Range, 3> ranges;

    for (const auto& index : rhs) {
      ranges.emplace_back(index, index + 1);
    }

    return *this += MultidimensionalRange(std::move(ranges));
  }

  IndexSet::Impl& IndexSet::Impl::operator+=(const MultidimensionalRange& rhs)
  {
    if (!initialized) {
      allowedRank = rhs.rank();
      initialized = true;
    }

    assert(rhs.rank() == allowedRank && "Incompatible rank");

    // We must add only the non-existing points
    std::queue<MultidimensionalRange> nonOverlappingRanges;

    if (root == nullptr) {
      nonOverlappingRanges.push(rhs);
    } else {
      std::vector<MultidimensionalRange> current;
      current.push_back(rhs);

      for (const auto& range : *this) {
        std::vector<MultidimensionalRange> next;

        for (const auto& curr : current) {
          for (auto& diff : curr.subtract(range)) {
            if (overlaps(diff)) {
              next.push_back(std::move(diff));
            } else {
              // For safety, also check that all the range we are going to add do belongs
              // to the original range.
              assert(rhs.contains(diff));
              nonOverlappingRanges.push(std::move(diff));
            }
          }
        }

        current = std::move(next);
      }
    }

    while (!nonOverlappingRanges.empty()) {
      auto& range = nonOverlappingRanges.front();

      // Check that all the range do not overlap the existing points
      assert(!overlaps(range));

      if (root == nullptr) {
        root = std::make_unique<Node>(nullptr, range);
        root->add(std::move(range));
      } else {
        // Find position for the new record
        auto node = chooseLeaf(range);

        // Add the record to the leaf node
        node->add(std::move(range));

        // Merge the adjacent ranges
        llvm::sort(node->values);
        merge(node->values);

        if (!node->isRoot() && node->fanOut() < minElements) {
          while (!node->isRoot() && node->fanOut() < minElements) {
            std::stack<Node*> nodes;
            nodes.push(node);

            while (!nodes.empty()) {
              auto current = nodes.top();
              nodes.pop();

              for (auto& child : current->children) {
                nodes.push(child.get());
              }

              for (auto& value : current->values) {
                nonOverlappingRanges.push(std::move(value));
              }
            }

            for (const auto& value : node->values) {
              nonOverlappingRanges.push(value);
            }

            auto parent = node->parent;
            assert(parent != nullptr);

            std::vector<std::unique_ptr<Node>> newChildren;

            for (auto& child : node->parent->children) {
              if (child.get() != node) {
                newChildren.push_back(std::move(child));
              }
            }

            parent->children = std::move(newChildren);
            parent->recalcBoundary();
            node = parent;
          }
        } else if (node->fanOut() > maxElements) {
          // Propagate node splits
          while (node->fanOut() > maxElements) {
            auto newNodes = splitNode(*node);

            if (node->isRoot()) {
              // If node split propagation caused the root to split, then
              // create a new root whose children are the two resulting nodes.
              auto rootBoundary = getMBR(newNodes.first->getBoundary(), newNodes.second->getBoundary());

              root = std::make_unique<Node>(nullptr, rootBoundary);

              newNodes.first->parent = root.get();
              newNodes.second->parent = root.get();

              root->add(std::move(newNodes.first));
              root->add(std::move(newNodes.second));

              node = root.get();
              break;
            }

            *node = std::move(*newNodes.first);

            for (auto& child : node->children) {
              child->parent = node;
            }

            node->parent->add(std::move(newNodes.second));

            // Propagate changes upward
            node = node->parent;
          }
        }

        while (node != nullptr) {
          node->recalcBoundary();
          node = node->parent;
        }
      }

      nonOverlappingRanges.pop();

      // Check that all the invariants are respected
      assert(isValid());
    }

    return *this;
  }

  IndexSet::Impl& IndexSet::Impl::operator+=(const IndexSet::Impl& rhs)
  {
    for (const auto& range : rhs) {
      *this += range;
    }

    return *this;
  }

  IndexSet::Impl IndexSet::Impl::operator+(const Point& rhs) const
  {
    IndexSet::Impl result(*this);
    result += rhs;
    return result;
  }

  IndexSet::Impl IndexSet::Impl::operator+(const MultidimensionalRange& rhs) const
  {
    IndexSet::Impl result(*this);
    result += rhs;
    return result;
  }

  IndexSet::Impl IndexSet::Impl::operator+(const IndexSet::Impl& rhs) const
  {
    IndexSet::Impl result(*this);
    result += rhs;
    return result;
  }

  IndexSet::Impl& IndexSet::Impl::operator-=(const MultidimensionalRange& rhs)
  {
    if (!initialized) {
      allowedRank = rhs.rank();
      initialized = true;
    }

    assert(rhs.rank() == allowedRank && "Incompatible rank");

    if (root == nullptr) {
      return *this;
    }

    if (!root->getBoundary().overlaps(rhs)) {
      return *this;
    }

    auto oldRoot = std::move(root);
    root = nullptr;

    std::stack<const Node*> nodes;
    nodes.push(oldRoot.get());

    while (!nodes.empty()) {
      auto node = nodes.top();
      nodes.pop();

      for (const auto& child : node->children) {
        nodes.push(child.get());
      }

      for (const auto& value : node->values) {
        for (const auto& difference : value.subtract(rhs)) {
          *this += difference;
        }
      }
    }

    return *this;
  }

  IndexSet::Impl& IndexSet::Impl::operator-=(const IndexSet::Impl& rhs)
  {
    for (const auto& range : rhs) {
      *this -= range;
    }

    return *this;
  }

  IndexSet::Impl IndexSet::Impl::operator-(const MultidimensionalRange& rhs) const
  {
    IndexSet::Impl result(*this);
    result -= rhs;
    return result;
  }

  IndexSet::Impl IndexSet::Impl::operator-(const IndexSet::Impl& rhs) const
  {
    IndexSet::Impl result(*this);
    result -= rhs;
    return result;
  }

  bool IndexSet::Impl::empty() const
  {
    return root == nullptr;
  }

  size_t IndexSet::Impl::rank() const
  {
    assert(initialized);
    return allowedRank;
  }

  size_t IndexSet::Impl::flatSize() const
  {
    size_t result = 0;

    for (const auto& range : *this) {
      result += range.flatSize();
    }

    return result;
  }

  void IndexSet::Impl::clear()
  {
    root = nullptr;
  }

  IndexSet::Impl::const_iterator IndexSet::Impl::begin() const
  {
    return IndexSet::Impl::Iterator::begin(*this);
  }

  IndexSet::Impl::const_iterator IndexSet::Impl::end() const
  {
    return IndexSet::Impl::Iterator::end(*this);
  }

  bool IndexSet::Impl::contains(const Point& other) const
  {
    return contains(MultidimensionalRange(other));
  }

  bool IndexSet::Impl::contains(const MultidimensionalRange& other) const
  {
    if (root == nullptr) {
      return false;
    }

    std::queue<MultidimensionalRange> remainingRanges;
    remainingRanges.push(other);
    bool changesDetected = true;

    while (!remainingRanges.empty() && changesDetected) {
      changesDetected = false;
      const auto& range = remainingRanges.front();

      std::stack<const Node*> nodes;

      if (root->getBoundary().overlaps(range)) {
        nodes.push(root.get());
      }

      while (!nodes.empty() && !changesDetected) {
        auto node = nodes.top();
        nodes.pop();

        for (const auto& child : node->children) {
          if (child->getBoundary().overlaps(range)) {
            nodes.push(child.get());
          }
        }

        for (const auto& value : node->values) {
          if (value.overlaps(range)) {
            for (const auto& difference : range.subtract(value)) {
              remainingRanges.push(difference);
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

  bool IndexSet::Impl::contains(const IndexSet::Impl& other) const
  {
    for (const auto& range : other) {
      if (!contains(range)) {
        return false;
      }
    }

    return true;
  }

  bool IndexSet::Impl::overlaps(const MultidimensionalRange& other) const
  {
    if (root == nullptr) {
      return false;
    }

    std::stack<const Node*> nodes;

    if (root->getBoundary().overlaps(other)) {
      nodes.push(root.get());
    }

    while (!nodes.empty()) {
      auto node = nodes.top();
      nodes.pop();

      for (const auto& child : node->children) {
        if (child->getBoundary().overlaps(other)) {
          nodes.push(child.get());
        }
      }

      for (const auto& value : node->values) {
        if (value.overlaps(other)) {
          return true;
        }
      }
    }

    return false;
  }

  bool IndexSet::Impl::overlaps(const IndexSet::Impl& other) const
  {
    return llvm::any_of(other, [&](const auto& range) {
      return overlaps(range);
    });
  }

  IndexSet::Impl IndexSet::Impl::intersect(const MultidimensionalRange& other) const
  {
    IndexSet::Impl result;
    result.initialized = initialized;
    result.allowedRank = allowedRank;

    if (root == nullptr) {
      return result;
    }

    std::stack<const Node*> nodes;

    if (root->getBoundary().overlaps(other)) {
      nodes.push(root.get());
    } else {
      return result;
    }

    MultidimensionalRange intersection = other;

    while (!nodes.empty()) {
      auto node = nodes.top();
      nodes.pop();

      for (const auto& child : node->children) {
        if (child->getBoundary().overlaps(intersection)) {
          nodes.push(child.get());
        }
      }

      for (const auto& value : node->values) {
        if (value.overlaps(intersection)) {
          intersection = intersection.intersect(value);
        }
      }
    }

    result += intersection;
    return result;
  }

  IndexSet::Impl IndexSet::Impl::intersect(const IndexSet::Impl& other) const
  {
    IndexSet::Impl result;
    result.initialized = initialized;
    result.allowedRank = allowedRank;

    if (root == nullptr) {
      return result;
    }

    for (const auto& range : other) {
      result += this->intersect(range);
    }

    return result;
  }

  IndexSet::Impl IndexSet::Impl::complement(const MultidimensionalRange& other) const
  {
    if (root == nullptr) {
      return IndexSet::Impl(other);
    }

    std::vector<MultidimensionalRange> result;

    std::vector<MultidimensionalRange> current;
    current.push_back(other);

    for (const auto& range : *this) {
      std::vector<MultidimensionalRange> next;

      for (const auto& curr : current) {
        for (const auto& diff : curr.subtract(range)) {
          if (overlaps(diff)) {
            next.push_back(std::move(diff));
          } else {
            result.push_back(std::move(diff));
          }
        }
      }

      current = next;
    }

    return IndexSet::Impl(result);
  }

  const Node* IndexSet::Impl::getRoot() const
  {
    return root.get();
  }

  Node* IndexSet::Impl::chooseLeaf(const MultidimensionalRange& entry) const
  {
    Node* node = root.get();

    while (!node->isLeaf()) {
      // Choose the child that needs the least enlargement to include the new element
      std::vector<std::pair<MultidimensionalRange, size_t>> candidateMBRs;

      for (const auto& child : node->children) {
        auto mbr = getMBR(child->getBoundary(), entry);
        size_t flatDifference = mbr.flatSize() - child->getFlatSize();
        candidateMBRs.emplace_back(std::move(mbr), flatDifference);
      }

      // Select the child that needs the least enlargement to include the new entry.
      // Resolve ties by choosing the node with the rectangle with the smallest area.
      auto enumeratedCandidates = llvm::enumerate(candidateMBRs);

      auto it = std::min_element(
          enumeratedCandidates.begin(), enumeratedCandidates.end(),
          [](const auto& first, const auto& second) {
            if (first.value().second == second.value().second) {
              return first.value().first.flatSize() < second.value().first.flatSize();
            }

            return first.value().second < second.value().second;
          });

      node = node->children[(*it).index()].get();
    }

    return node;
  }

  std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>> IndexSet::Impl::splitNode(Node& node)
  {
    if (node.isLeaf()) {
      return splitNode<MultidimensionalRange>(node, [](Node& node) -> std::vector<MultidimensionalRange>& {
        return node.values;
      });
    } else {
      auto result = splitNode<std::unique_ptr<Node>>(node, [](Node& node) -> std::vector<std::unique_ptr<Node>>& {
        return node.children;
      });

      // We need to fix the parent address
      for (auto& child : result.first->children) {
        child->parent = result.first.get();
      }

      for (auto& child : result.second->children) {
        child->parent = result.second.get();
      }

      return result;
    }
  }

  bool IndexSet::Impl::isValid() const
  {
    if (!checkParentRelationships(*this)) {
      std::cerr << "IndexSet: parents hierarchy invariant failure" << std::endl;
      return false;
    }

    if (!checkMBRsInvariant(*this)) {
      std::cerr << "IndexSet: MBRs invariant failure" << std::endl;
      return false;
    }

    if (!checkFanOutInvariant(*this)) {
      std::cerr << "IndexSet: fan-out invariant failure" << std::endl;
      return false;
    }

    return true;
  }

  IndexSet::Impl::Iterator::Iterator(const Node* root)
    : node(root),
      valueIndex(0)
  {
    if (root != nullptr) {
      nodes.push(root);

      while (!nodes.top()->isLeaf()) {
        auto current = nodes.top();
        nodes.pop();
        const auto& children = current->children;

        for (size_t i = 0, e = children.size(); i < e; ++i) {
          nodes.push(children[e - i - 1].get());
        }
      }

      node = nodes.top();
    }
  }

  IndexSet::Impl::Iterator IndexSet::Impl::Iterator::begin(const IndexSet::Impl& indexSet)
  {
    return Iterator(indexSet.root.get());
  }

  IndexSet::Impl::Iterator IndexSet::Impl::Iterator::end(const IndexSet::Impl& indexSet)
  {
    return Iterator(nullptr);
  }

  bool IndexSet::Impl::Iterator::operator==(const Iterator& it) const
  {
    return node == it.node && valueIndex == it.valueIndex;
  }

  bool IndexSet::Impl::Iterator::operator!=(const Iterator& it) const
  {
    return node != it.node || valueIndex != it.valueIndex;
  }

  IndexSet::Impl::Iterator& IndexSet::Impl::Iterator::operator++()
  {
    ++valueIndex;

    if (valueIndex == node->values.size()) {
      fetchNextLeaf();
      valueIndex = 0;
    }

    return *this;
  }

  IndexSet::Impl::Iterator IndexSet::Impl::Iterator::operator++(int)
  {
    auto temp = *this;
    ++(*this);
    return temp;
  }

  IndexSet::Impl::Iterator::reference IndexSet::Impl::Iterator::operator*() const
  {
    assert(node != nullptr);
    assert(node->isLeaf());
    return node->values[valueIndex];
  }

  void IndexSet::Impl::Iterator::fetchNextLeaf()
  {
    nodes.pop();

    while (!nodes.empty() && !nodes.top()->isLeaf()) {
      auto current = nodes.top();
      nodes.pop();
      const auto& children = current->children;

      for (size_t i = 0, e = children.size(); i < e; ++i) {
        nodes.push(children[e - i - 1].get());
      }
    }

    if (nodes.empty()) {
      node = nullptr;
    } else {
      node = nodes.top();
    }
  }
}

namespace marco::modeling
{
  class IndexSet::Iterator::Impl
  {
    public:
      using iterator_category = std::input_iterator_tag;
      using value_type = MultidimensionalRange;
      using difference_type = std::ptrdiff_t;
      using pointer = const MultidimensionalRange*;
      using reference = const MultidimensionalRange&;

      static Impl begin(const IndexSet& indexSet);

      static Impl end(const IndexSet& indexSet);

      bool operator==(const Impl& it) const;

      bool operator!=(const Impl& it) const;

      Impl& operator++();

      Impl operator++(int);

      reference operator*() const;

    private:
      Impl(IndexSet::Impl::Iterator impl);

      IndexSet::Impl::Iterator impl;
  };

  IndexSet::Iterator::Impl::Impl(IndexSet::Impl::Iterator impl)
    : impl(std::move(impl))
  {
  }

  IndexSet::Iterator::Impl IndexSet::Iterator::Impl::begin(const IndexSet& indexSet)
  {
    return Impl(indexSet.impl->begin());
  }

  IndexSet::Iterator::Impl IndexSet::Iterator::Impl::end(const IndexSet& indexSet)
  {
    return Impl(indexSet.impl->end());
  }

  bool IndexSet::Iterator::Impl::operator==(const IndexSet::Iterator::Impl& it) const
  {
    return impl == it.impl;
  }

  bool IndexSet::Iterator::Impl::operator!=(const IndexSet::Iterator::Impl& it) const
  {
    return impl != it.impl;
  }

  IndexSet::Iterator::Impl& IndexSet::Iterator::Impl::operator++()
  {
    ++impl;
    return *this;
  }

  IndexSet::Iterator::Impl IndexSet::Iterator::Impl::operator++(int)
  {
    auto temp = *this;
    ++impl;
    return temp;
  }

  IndexSet::Iterator::Impl::reference IndexSet::Iterator::Impl::operator*() const
  {
    return *impl;
  }
}

namespace marco::modeling
{
  IndexSet::Iterator::Iterator(std::unique_ptr<Impl> impl)
    : impl(std::move(impl))
  {
  }

  IndexSet::Iterator::Iterator(const IndexSet::Iterator& other)
    : impl(std::make_unique<Impl>(*other.impl))
  {
  }

  IndexSet::Iterator::Iterator(IndexSet::Iterator&& other) = default;

  IndexSet::Iterator::~Iterator() = default;

  IndexSet::Iterator& IndexSet::Iterator::operator=(const IndexSet::Iterator& other)
  {
    IndexSet::Iterator result(other);
    swap(*this, result);
    return *this;
  }

  void swap(IndexSet::Iterator& first, IndexSet::Iterator& second)
  {
    using std::swap;
    swap(first.impl, second.impl);
  }

  IndexSet::Iterator IndexSet::Iterator::begin(const IndexSet& indexSet)
  {
    return Iterator(std::make_unique<Impl>(Impl::begin(indexSet)));
  }

  IndexSet::Iterator IndexSet::Iterator::end(const IndexSet& indexSet)
  {
    return Iterator(std::make_unique<Impl>(Impl::end(indexSet)));
  }

  bool IndexSet::Iterator::operator==(const IndexSet::Iterator& it) const
  {
    return *impl == *it.impl;
  }

  bool IndexSet::Iterator::operator!=(const IndexSet::Iterator& it) const
  {
    return *impl != *it.impl;
  }

  IndexSet::Iterator& IndexSet::Iterator::operator++()
  {
    ++(*impl);
    return *this;
  }

  IndexSet::Iterator IndexSet::Iterator::operator++(int)
  {
    auto temp = *this;
    ++(*impl);
    return temp;
  }

  IndexSet::Iterator::reference IndexSet::Iterator::operator*() const
  {
    return **impl;
  }

  IndexSet::IndexSet(std::unique_ptr<IndexSet::Impl> impl)
    : impl(std::move(impl))
  {
  }

  IndexSet::IndexSet()
      : IndexSet(std::make_unique<Impl>())
  {
  }

  IndexSet::IndexSet(llvm::ArrayRef<Point> points)
    : IndexSet(std::make_unique<Impl>(points))
  {
  }

  IndexSet::IndexSet(llvm::ArrayRef<MultidimensionalRange> ranges)
    : IndexSet(std::make_unique<Impl>(ranges))
  {
  }

  IndexSet::IndexSet(const IndexSet& other)
    : impl(std::make_unique<Impl>(*other.impl))
  {
  }

  IndexSet::IndexSet(IndexSet&& other) = default;

  IndexSet::~IndexSet() = default;

  IndexSet& IndexSet::operator=(const IndexSet& other)
  {
    IndexSet result(other);
    swap(*this, result);
    return *this;
  }

  void swap(IndexSet& first, IndexSet& second)
  {
    using std::swap;
    swap(first.impl, second.impl);
  }

  bool IndexSet::operator==(const Point& rhs) const
  {
    return *impl == rhs;
  }

  bool IndexSet::operator==(const MultidimensionalRange& rhs) const
  {
    return *impl == rhs;
  }

  bool IndexSet::operator==(const IndexSet& rhs) const
  {
    return *impl == *rhs.impl;
  }

  bool IndexSet::operator!=(const Point& rhs) const
  {
    return *impl != rhs;
  }

  bool IndexSet::operator!=(const MultidimensionalRange& rhs) const
  {
    return *impl != rhs;
  }

  bool IndexSet::operator!=(const IndexSet& rhs) const
  {
    return *impl != *rhs.impl;
  }

  IndexSet& IndexSet::operator+=(const Point& rhs)
  {
    *impl += rhs;
    return *this;
  }

  IndexSet& IndexSet::operator+=(const MultidimensionalRange& rhs)
  {
    *impl += rhs;
    return *this;
  }

  IndexSet& IndexSet::operator+=(const IndexSet& rhs)
  {
    *impl += *rhs.impl;
    return *this;
  }

  IndexSet IndexSet::operator+(const Point& rhs) const
  {
    IndexSet result(*this);
    result += rhs;
    return result;
  }

  IndexSet IndexSet::operator+(const MultidimensionalRange& rhs) const
  {
    IndexSet result(*this);
    result += rhs;
    return result;
  }

  IndexSet IndexSet::operator+(const IndexSet& rhs) const
  {
    IndexSet result(*this);
    result += rhs;
    return result;
  }

  IndexSet& IndexSet::operator-=(const MultidimensionalRange& rhs)
  {
    *impl -= rhs;
    return *this;
  }

  IndexSet& IndexSet::operator-=(const IndexSet& rhs)
  {
    *impl -= *rhs.impl;
    return *this;
  }

  IndexSet IndexSet::operator-(const MultidimensionalRange& rhs) const
  {
    IndexSet result(*this);
    result -= rhs;
    return result;
  }

  IndexSet IndexSet::operator-(const IndexSet& rhs) const
  {
    IndexSet result(*this);
    result -= rhs;
    return result;
  }

  bool IndexSet::empty() const
  {
    return impl->empty();
  }

  size_t IndexSet::rank() const
  {
    return impl->rank();
  }

  size_t IndexSet::flatSize() const
  {
    return impl->flatSize();
  }

  void IndexSet::clear()
  {
    impl->clear();
  }

  // IndexSet::const_iterator IndexSet::begin() const
  // {
  //   return Iterator::begin(*this);
  // }

  // IndexSet::const_iterator IndexSet::end() const
  // {
  //   return Iterator::end(*this);
  // }

  llvm::iterator_range<IndexSet::const_iterator> IndexSet::getRanges() const
  {
    return llvm::make_range(Iterator::begin(*this),Iterator::end(*this));
  }

  bool IndexSet::contains(const Point& other) const
  {
    return impl->contains(other);
  }

  bool IndexSet::contains(const MultidimensionalRange& other) const
  {
    return impl->contains(other);
  }

  bool IndexSet::contains(const IndexSet& other) const
  {
    return impl->contains(*other.impl);
  }

  bool IndexSet::overlaps(const MultidimensionalRange& other) const
  {
    return impl->overlaps(other);
  }

  bool IndexSet::overlaps(const IndexSet& other) const
  {
    return impl->overlaps(*other.impl);
  }

  IndexSet IndexSet::intersect(const MultidimensionalRange& other) const
  {
    return IndexSet(std::make_unique<Impl>(impl->intersect(other)));
  }

  IndexSet IndexSet::intersect(const IndexSet& other) const
  {
    return IndexSet(std::make_unique<Impl>(impl->intersect(*other.impl)));
  }

  IndexSet IndexSet::complement(const MultidimensionalRange& other) const
  {
    return IndexSet(std::make_unique<Impl>(impl->complement(other)));
  }

  std::ostream& operator<<(std::ostream& os, const IndexSet& obj)
  {
    return os << *obj.impl;
  }
}
