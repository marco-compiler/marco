#ifndef MARCO_MODELING_RTREE_H
#define MARCO_MODELING_RTREE_H

#include "marco/Modeling/MultidimensionalRange.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace marco::modeling {
// This class must be specialized for the type of the elements that are inserted
// in the R-Tree.
template <typename T>
struct RTreeInfo {
  // static MultidimensionalRange getShape(const T &val);
  //   Get the shape of the object.
  //   A reference to a precomputed shape can also be returned, if available.
  //
  // static bool isEqual(const T &first, const T &second);
  //   Check whether two values are equal.

  using Info = typename T::UnknownTypeError;
};

/// R-Tree information specialization for the MultidimensionalRange class.
template <>
struct RTreeInfo<MultidimensionalRange> {
  static const MultidimensionalRange &
  getShape(const MultidimensionalRange &val);

  static bool isEqual(const MultidimensionalRange &first,
                      const MultidimensionalRange &second);

  static void dump(llvm::raw_ostream &os, const MultidimensionalRange &val);
};

namespace r_tree::impl {
template <typename T>
decltype(auto) getShape(const T &obj) {
  return RTreeInfo<T>::getShape(obj);
}

/// Wrapper for the object to be stored. In the most general case, it allows
/// computing the boundary of the object only once, and store it together with
/// the actual object.
template <typename T>
class Object {
  T value;
  MultidimensionalRange boundary;

public:
  Object(T value, MultidimensionalRange boundary)
      : value(std::move(value)), boundary(std::move(boundary)) {}

  Object(T value)
      : value(std::move(value)), boundary(impl::getShape(this->value)) {}

  T &operator*() { return value; }

  const T &operator*() const { return value; }

  T *operator->() { return &value; }

  const T *operator->() const { return &value; }

  const MultidimensionalRange &getBoundary() const { return boundary; }
};

/// Object specialization for the multidimensional range, whose boundary exactly
/// matches the value.
template <>
class Object<MultidimensionalRange> {
  MultidimensionalRange value;

public:
  Object(MultidimensionalRange value) : value(std::move(value)) {}

  MultidimensionalRange &operator*() { return value; }

  const MultidimensionalRange &operator*() const { return value; }

  MultidimensionalRange *operator->() { return &value; }

  const MultidimensionalRange *operator->() const { return &value; }

  const MultidimensionalRange &getBoundary() const { return value; }
};

template <typename T>
const MultidimensionalRange &getShape(const Object<T> &obj) {
  return obj.getBoundary();
}

/// Get the minimum bounding (multidimensional) range for two ranges.
MultidimensionalRange getMBR(const MultidimensionalRange &first,
                             const MultidimensionalRange &second);

/// Get the minimum bounding range for several elements.
template <typename T>
MultidimensionalRange getMBR(llvm::ArrayRef<T> objects) {
  assert(!objects.empty());
  MultidimensionalRange result(getShape(objects[0]));

  for (size_t i = 1, e = objects.size(); i < e; ++i) {
    result = getMBR(result, getShape(objects[i]));
  }

  assert(llvm::all_of(objects, [&](const auto &obj) {
    return result.contains(getShape(obj));
  }));

  return result;
}

/// A node of the R-Tree.
template <typename T>
class Node {
  /// Parent node in the tree.
  Node *parent;

  /// MBR containing all the children / objects.
  MultidimensionalRange boundary;

  /// The flat size of the MBR.
  /// It is stored internally to avoid recomputing it all the times (an
  /// operation that has an O(d) cost, with d equal to the rank).
  unsigned int coveredSpaceSize;

public:
  llvm::SmallVector<std::unique_ptr<Node>> children;
  llvm::SmallVector<Object<T>> objects;

  Node(Node *parent, const MultidimensionalRange &boundary)
      : parent(parent), boundary(boundary), coveredSpaceSize(boundary.size()) {
    assert(this != parent && "A node can't be the parent of itself");
  }

  Node(const Node &other)
      : parent(other.parent), boundary(other.boundary),
        coveredSpaceSize(other.coveredSpaceSize) {
    for (const auto &child : other.children) {
      add(std::make_unique<Node>(*child));
    }

    for (const Object<T> &object : other.objects) {
      add(object);
    }
  }

  ~Node() = default;

  Node &operator=(Node &&other) = default;

  Node *getParent() const { return parent; }

  void setParent(Node *newParent) { parent = newParent; }

  bool isRoot() const { return parent == nullptr; }

  bool isLeaf() const {
    auto result = children.empty();
    assert(result || objects.empty());
    return result;
  }

  const MultidimensionalRange &getBoundary() const { return boundary; }

private:
  /// Compute the boundary including all the children or objects of the node.
  MultidimensionalRange computeBoundary() const {
    assert(fanOut() > 0);

    if (isLeaf()) {
      return impl::getMBR(llvm::ArrayRef(objects));
    }

    return impl::getMBR(llvm::ArrayRef(children));
  }

public:
  /// Recalculate the MBR containing all the objects / children.
  /// Return true if the boundary changed, false otherwise.
  bool recalcBoundary() {
    MultidimensionalRange oldBoundary(std::move(boundary));
    boundary = computeBoundary();

    // Update the size of the covered space.
    coveredSpaceSize = boundary.size();

    return boundary != oldBoundary;
  }

  /// Get the multidimensional space (area, volume, etc.) covered by
  /// the MBR.
  size_t getCoveredSpaceSize() const { return coveredSpaceSize; }

  /// Get the number of children or objects.
  size_t fanOut() const {
    if (isLeaf()) {
      return objects.size();
    }

    return children.size();
  }

  /// Add a child node.
  void add(std::unique_ptr<Node> child) {
    assert(objects.empty());
    child->parent = this;
    children.push_back(std::move(child));
  }

  /// Add an object.
  void add(Object<T> value) {
    assert(children.empty());
    objects.push_back(std::move(value));
  }

  /// Add an object.
  void add(T value) { objects.emplace_back(std::move(value)); }
};

template <typename T>
const MultidimensionalRange &getShape(const std::unique_ptr<Node<T>> &node) {
  return node->getBoundary();
}

/// Select two entries to be the first elements of the new groups.
template <typename T>
std::pair<size_t, size_t> pickSeeds(llvm::ArrayRef<T> entries) {
  std::vector<std::tuple<size_t, size_t, size_t>> candidates;

  for (size_t i = 0; i < entries.size(); ++i) {
    for (size_t j = i + 1; j < entries.size(); ++j) {
      const MultidimensionalRange &firstShape = getShape(entries[i]);
      const MultidimensionalRange &secondShape = getShape(entries[j]);
      MultidimensionalRange mbr = getMBR(firstShape, secondShape);

      size_t mbrSize = mbr.size();
      size_t firstEntryBoundarySize = firstShape.size();
      size_t secondEntryBoundarySize = secondShape.size();

      size_t difference = 0;

      if (firstEntryBoundarySize + secondEntryBoundarySize < mbrSize) {
        difference = mbrSize - firstEntryBoundarySize - secondEntryBoundarySize;
      }

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
template <typename T, typename ShapedT>
size_t pickNext(llvm::ArrayRef<ShapedT> entries,
                const llvm::SmallSetVector<size_t, 16> &remainingEntries,
                const Node<T> &firstNode, const Node<T> &secondNode) {
  assert(!entries.empty());

  // Determine the cost of putting each entry in each group.
  std::vector<std::pair<size_t, size_t>> costs;

  for (size_t remainingEntryIndex : remainingEntries) {
    const MultidimensionalRange &shape = getShape(entries[remainingEntryIndex]);

    auto d1 = getMBR(shape, firstNode.getBoundary()).size() -
              firstNode.getBoundary().size();

    auto d2 = getMBR(shape, secondNode.getBoundary()).size() -
              secondNode.getBoundary().size();

    auto minMax = std::minmax(d1, d2);
    costs.emplace_back(remainingEntryIndex, minMax.second - minMax.first);
  }

  // Find the entry with the greatest preference for one group.
  assert(!costs.empty());

  auto it = std::max_element(costs.begin(), costs.end(),
                             [](const auto &first, const auto &second) {
                               return first.second < second.second;
                             });

  return it->first;
}

template <typename T, typename U>
std::pair<std::unique_ptr<Node<T>>, std::unique_ptr<Node<T>>> splitNode(
    Node<T> &node, const size_t minElements,
    llvm::function_ref<llvm::SmallVectorImpl<U> &(Node<T> &)> containerFn) {
  auto seeds = pickSeeds(llvm::ArrayRef(containerFn(node)));

  // The position of the elements yet to be relocated.
  llvm::SmallSetVector<size_t, 16> remainingObjects;

  for (size_t i = 0, e = containerFn(node).size(); i < e; ++i) {
    remainingObjects.insert(i);
  }

  auto moveValueFn = [&](Node<T> &destination, size_t objectIndex) {
    if (remainingObjects.contains(objectIndex)) {
      containerFn(destination)
          .push_back(std::move(containerFn(node)[objectIndex]));

      remainingObjects.remove(objectIndex);
    }
  };

  auto moveRemainingFn = [&](Node<T> &destination) {
    auto &sourceContainer = containerFn(node);
    auto &destinationContainer = containerFn(destination);

    for (size_t remainingObjectIndex : remainingObjects) {
      destinationContainer.push_back(
          std::move(sourceContainer[remainingObjectIndex]));
    }

    remainingObjects.clear();
  };

  auto firstNew = std::make_unique<Node<T>>(
      node.getParent(), getShape(containerFn(node)[seeds.first]));

  moveValueFn(*firstNew, seeds.first);

  auto secondNew = std::make_unique<Node<T>>(
      node.getParent(), getShape(containerFn(node)[seeds.second]));

  moveValueFn(*secondNew, seeds.second);

  // Keep processing the objects until all of them have been assigned to the
  // new nodes.
  while (!remainingObjects.empty()) {
    size_t remaining = remainingObjects.size();

    // If one group has so few entries that all the rest must be assigned to
    // it in order for it to have the minimum number of elements, then assign
    // them and stop.

    if (containerFn(*firstNew).size() + remaining == minElements) {
      moveRemainingFn(*firstNew);
      break;
    }

    if (containerFn(*secondNew).size() + remaining == minElements) {
      moveRemainingFn(*secondNew);
      break;
    }

    // Choose the next entry to assign.
    auto next = pickNext(llvm::ArrayRef(containerFn(node)), remainingObjects,
                         *firstNew, *secondNew);

    assert(remainingObjects.contains(next) && "Already relocated index");

    // Add it to the group whose covering rectangle will have to be
    // enlarged least to accommodate it. Resolve ties by adding the entry to
    // the group with smaller area, then to the one with fewer entries, then
    // to either of them.

    auto firstEnlargement =
        getMBR(getShape(containerFn(node)[next]), firstNew->getBoundary())
            .size() -
        firstNew->getCoveredSpaceSize();

    auto secondEnlargement =
        getMBR(getShape(containerFn(node)[next]), secondNew->getBoundary())
            .size() -
        secondNew->getCoveredSpaceSize();

    if (firstEnlargement == secondEnlargement) {
      size_t firstSpaceSize = firstNew->getCoveredSpaceSize();
      size_t secondSpaceSize = secondNew->getCoveredSpaceSize();

      if (firstSpaceSize == secondSpaceSize) {
        size_t firstElementsAmount = containerFn(*firstNew).size();
        size_t secondElementsAmount = containerFn(*secondNew).size();

        if (firstElementsAmount <= secondElementsAmount) {
          moveValueFn(*firstNew, next);
        } else {
          moveValueFn(*secondNew, next);
        }
      } else if (firstSpaceSize < secondSpaceSize) {
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

template <typename T>
class ObjectIterator {
  llvm::SmallVector<const Node<T> *> nodes;
  const Node<T> *node;
  size_t objectIndex{0};

public:
  using iterator_category = std::input_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = const T *;
  using reference = const T &;

  explicit ObjectIterator(const Node<T> *root) : node(root) {
    if (root) {
      nodes.push_back(root);

      while (!nodes.back()->isLeaf()) {
        const Node<T> *current = nodes.pop_back_val();
        const auto &children = current->children;

        for (size_t i = 0, e = children.size(); i < e; ++i) {
          nodes.push_back(children[e - i - 1].get());
        }
      }

      node = nodes.back();
    }
  }

  bool operator==(const ObjectIterator &other) const {
    return node == other.node && objectIndex == other.objectIndex;
  }

  bool operator!=(const ObjectIterator &other) const {
    return !(*this == other);
  }

  ObjectIterator &operator++() {
    ++objectIndex;

    if (objectIndex == node->objects.size()) {
      fetchNextLeaf();
      objectIndex = 0;
    }

    return *this;
  }

  ObjectIterator operator++(int) {
    ObjectIterator result(*this);
    ++(*this);
    return result;
  }

  reference operator*() const {
    assert(node != nullptr);
    assert(node->isLeaf());
    return *node->objects[objectIndex];
  }

private:
  void fetchNextLeaf() {
    nodes.pop_back();

    while (!nodes.empty() && !nodes.back()->isLeaf()) {
      const Node<T> *current = nodes.pop_back_val();
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
};

template <typename Derived, typename T>
class RTreeCRTP {
protected:
  using Node = r_tree::impl::Node<T>;
  using Object = r_tree::impl::Object<T>;
  using const_object_iterator = r_tree::impl::ObjectIterator<T>;

private:
  /// The minimum number of elements for each node (apart the root).
  size_t minElements{4};

  /// The maximum number of elements for each node.
  size_t maxElements{16};

  /// The root of the tree.
  std::unique_ptr<Node> root{nullptr};

  /// Whether a value has ever been inserted.
  bool initialized{false};

  /// The allowed rank for the boundaries of the objects.
  size_t allowedRank{0};

public:
  RTreeCRTP() = default;

  RTreeCRTP(const size_t minElements, const size_t maxElements)
      : minElements(minElements), maxElements(maxElements) {
    assert(maxElements > 1);
    assert(minElements >= 1);
    assert(minElements <= maxElements / 2);
  }

  RTreeCRTP(const RTreeCRTP &other)
      : minElements(other.minElements), maxElements(other.maxElements),
        root(other.root ? std::make_unique<Node>(*other.root) : nullptr),
        initialized(other.initialized), allowedRank(other.allowedRank) {}

  RTreeCRTP(RTreeCRTP &&other) noexcept = default;

  virtual ~RTreeCRTP() = default;

  RTreeCRTP &operator=(const RTreeCRTP &other) {
    minElements = other.minElements;
    maxElements = other.maxElements;
    root = other.root ? std::make_unique<Node>(*other.root) : nullptr;
    initialized = other.initialized;
    allowedRank = other.allowedRank;

    return *this;
  }

  RTreeCRTP &operator=(RTreeCRTP &&other) noexcept = default;

  virtual bool operator==(const RTreeCRTP &other) const {
    if (rank() != other.rank()) {
      return false;
    }

    return contains(other) && other.contains(*this);
  }

  virtual bool operator!=(const RTreeCRTP &other) const {
    return !(*this == other);
  }

  bool empty() const { return !root; }

  virtual size_t size() const {
    size_t count = 0;
    auto it = objectsBegin();

    while (it != objectsEnd()) {
      ++count;
      ++it;
    }

    return count;
  }

  size_t rank() const { return allowedRank; }

  void clear() { setRoot(nullptr); }

  Derived emptyCopy() const {
    Derived derived;
    static_cast<RTreeCRTP &>(derived).initialized = initialized;
    static_cast<RTreeCRTP &>(derived).allowedRank = allowedRank;
    return derived;
  }

protected:
  const_object_iterator objectsBegin() const {
    return const_object_iterator(getRoot());
  }

  const_object_iterator objectsEnd() const {
    return const_object_iterator(nullptr);
  }

public:
  virtual void insert(T obj) {
    const MultidimensionalRange &shape = getShape(obj);

    if (!initialized) {
      allowedRank = shape.rank();
      initialized = true;
    }

    assert(shape.rank() == allowedRank && "Incompatible rank");

    if (!getRoot()) {
      Node *newRoot = setRoot(std::make_unique<Node>(nullptr, shape));
      newRoot->add(std::move(obj));
      postObjectInsertionHook(*newRoot);
    } else {
      // Find position for the new object.
      Node *node = chooseLeaf(getRoot(), shape);

      // Add the object to the leaf node.
      node->add(std::move(obj));
      postObjectInsertionHook(*node);

      // Split and collapse nodes, if necessary.
      adjustTree(node);
    }

    // Check that all the invariants are respected.
    assert(isValid());
  }

  virtual void insert(const Derived &other) {
    for (const T &object :
         llvm::make_range(other.objectsBegin(), other.objectsEnd())) {
      insert(object);
    }
  }

  virtual void remove(const T &obj) {
    if (empty()) {
      return;
    }

    const MultidimensionalRange &shape = getShape(obj);
    llvm::SmallVector<Node *> overlappingLeafNodes;

    walkOverlappingLeafNodes(
        shape, [&](Node &node) { overlappingLeafNodes.push_back(&node); });

    for (Node *node : overlappingLeafNodes) {
      bool modified = false;
      llvm::SmallVector<Object> newObjects;

      for (Object &object : node->objects) {
        if (RTreeInfo<T>::isEqual(obj, *object)) {
          modified = true;
        } else {
          newObjects.push_back(std::move(object));
        }
      }

      node->objects = std::move(newObjects);

      if (modified) {
        adjustTree(node);
      }
    }
  }

  virtual void remove(const Derived &other) {
    if (empty() || other.empty()) {
      return;
    }

    using OverlappingNodePair = std::pair<const Node *, const Node *>;
    llvm::SmallVector<OverlappingNodePair> overlappingNodePairs;
    llvm::SmallVector<std::reference_wrapper<const Object>> overlappingObjects;

    const Node *lhsRoot = getRoot();
    const Node *rhsRoot = other.getRoot();

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
          for (const Object &lhsObject : lhs->objects) {
            for (const Object &rhsObject : rhs->objects) {
              if (lhsObject.getBoundary().overlaps(rhsObject.getBoundary())) {
                overlappingObjects.emplace_back(rhsObject);
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

    for (const Object &overlappingObject : overlappingObjects) {
      remove(*overlappingObject);
    }

    assert(isValid());
  }

  virtual bool contains(const T &other) const {
    if (empty()) {
      return false;
    }

    const MultidimensionalRange &shape = getShape(other);
    llvm::SmallVector<const Node *> nodes;

    if (getRoot()->getBoundary().overlaps(shape)) {
      nodes.push_back(getRoot());
    }

    while (!nodes.empty()) {
      const Node *node = nodes.pop_back_val();

      for (const auto &child : node->children) {
        if (child->getBoundary().overlaps(shape)) {
          nodes.push_back(child.get());
        }
      }

      for (const Object &object : node->objects) {
        if (object.getBoundary().overlaps(shape)) {
          if (RTreeInfo<T>::isEqual(other, *object)) {
            return true;
          }
        }
      }
    }

    return false;
  }

  virtual bool contains(const RTreeCRTP &other) const {
    if (other.empty()) {
      return true;
    }

    if (empty()) {
      return false;
    }

    const Node *lhsRoot = getRoot();
    const Node *rhsRoot = other.getRoot();

    llvm::SmallVector<const Node *> rhsNodes;
    llvm::SmallVector<const Node *> rhsLeaves;
    rhsNodes.push_back(rhsRoot);

    while (!rhsNodes.empty()) {
      const Node *node = rhsNodes.pop_back_val();

      if (!lhsRoot->getBoundary().contains(node->getBoundary())) {
        return false;
      }

      if (node->isLeaf()) {
        rhsLeaves.push_back(node);
      } else {
        for (const auto &child : node->children) {
          rhsNodes.push_back(child.get());
        }
      }
    }

    for (const Node *node : rhsLeaves) {
      for (const Object &object : node->objects) {
        if (!contains(*object)) {
          return false;
        }
      }
    }

    return true;
  }

protected:
  void setFromObjects(llvm::ArrayRef<T> objects) {
    setRoot(nullptr);

    if (objects.empty()) {
      return;
    }

    initialized = true;
    allowedRank = getShape(objects.front()).rank();

    // Create the tree structure.
    llvm::SmallVector<std::unique_ptr<Node>> nodes;

    if (objects.size() > maxElements) {
      size_t numOfGroups = objects.size() / minElements;
      size_t remainingElements = objects.size() % minElements;

      for (size_t i = 0; i < numOfGroups; ++i) {
        auto &node = nodes.emplace_back(std::make_unique<Node>(
            nullptr, getShape(objects[i * minElements])));

        for (size_t j = 0; j < minElements; ++j) {
          node->objects.emplace_back(std::move(objects[i * minElements + j]));
        }

        node->recalcBoundary();
      }

      for (size_t i = 0; i < remainingElements; ++i) {
        nodes[i]->objects.push_back(
            std::move(objects[numOfGroups * minElements + i]));
      }

      for (auto &node : nodes) {
        node->recalcBoundary();
      }
    } else {
      auto &node = nodes.emplace_back(
          std::make_unique<Node>(nullptr, getShape(objects.front())));

      node->objects.reserve(objects.size());
      llvm::append_range(node->objects, std::move(objects));
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

          parent->add(std::move(nodes[i + j]));
        }

        parent->recalcBoundary();
      }

      nodes = std::move(newNodes);
    }

    if (nodes.size() > 1) {
      auto parent = std::make_unique<Node>(nullptr, nodes[0]->getBoundary());

      for (auto &child : nodes) {
        parent->add(std::move(child));
      }

      parent->recalcBoundary();

      nodes.clear();
      nodes.push_back(std::move(parent));
    }

    setRoot(std::move(nodes.front()));
    assert(isValid() && "Inconsistent initialization of the R-Tree");
  }

  bool isInitialized() const { return initialized; }

  Node *getRoot() { return root.get(); }

  const Node *getRoot() const { return root.get(); }

  Node *setRoot(std::unique_ptr<Node> node) {
    root = std::move(node);

    if (root) {
      root->setParent(nullptr);
    }

    return root.get();
  }

  /// Update the boundary of a given node and propagate changes upwards in the
  /// tree.
  static void updateBoundaries(Node *node) {
    while (node && node->recalcBoundary()) {
      node = node->getParent();
    }
  }

  /// Select a leaf node in which to place a new entry.
  static Node *chooseLeaf(Node *searchRoot,
                          const MultidimensionalRange &entryBoundary) {
    Node *node = searchRoot;

    while (!node->isLeaf()) {
      node = chooseDestinationNode(node->children, entryBoundary);
    }

    return node;
  }

  /// Get the node that best accommodates the insertion of a new element with a
  /// given boundary.
  static Node *
  chooseDestinationNode(llvm::ArrayRef<std::unique_ptr<Node>> nodes,
                        const MultidimensionalRange &boundary) {
    assert(!nodes.empty());

    if (nodes.size() == 1) {
      return nodes.front().get();
    }

    // Choose the child that needs the least enlargement to include the new
    // element.
    std::vector<std::pair<MultidimensionalRange, size_t>> candidateMBRs;

    for (const auto &child : nodes) {
      auto mbr = getMBR(child->getBoundary(), boundary);
      size_t flatDifference = mbr.size() - child->getCoveredSpaceSize();
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
            return first.value().first.size() < second.value().first.size();
          }

          return first.value().second < second.value().second;
        });

    return nodes[(*it).index()].get();
  }

  /// Check and adjust the tree structure, starting from a given node, so that
  /// each node has between a minimum and maximum amount of children.
  void adjustTree(Node *node) {
    if (node->isRoot()) {
      if (node->fanOut() == 0) {
        setRoot(nullptr);
      } else if (node->fanOut() > maxElements) {
        splitNodeAndPropagate(node);
      } else {
        updateBoundaries(node);
      }
    } else if (node->fanOut() < minElements) {
      collapseNodeAndPropagate(node);
    } else if (node->fanOut() > maxElements) {
      splitNodeAndPropagate(node);
    } else {
      updateBoundaries(node);
    }
  }

  /// Split a node and its ancestors, if needed.
  void splitNodeAndPropagate(Node *node) {
    // Propagate node splits.
    while (node->fanOut() > maxElements) {
      auto newNodes = splitNode(*node);

      if (node->isRoot()) {
        // If node split propagation caused the root to split, then
        // create a new root whose children are the two resulting nodes.

        auto rootBoundary = getMBR(newNodes.first->getBoundary(),
                                   newNodes.second->getBoundary());

        node = setRoot(std::make_unique<Node>(nullptr, rootBoundary));
        node->add(std::move(newNodes.first));
        node->add(std::move(newNodes.second));
        break;
      }

      *node = std::move(*newNodes.first);

      for (auto &child : node->children) {
        child->setParent(node);
      }

      node->getParent()->add(std::move(newNodes.second));

      if (node->getParent()->fanOut() <= maxElements) {
        node->getParent()->recalcBoundary();
      }

      // Propagate changes upward.
      node = node->getParent();
    }

    // Fix the boundaries.
    if (!node->isRoot()) {
      updateBoundaries(node->getParent());
    }
  }

  /// Split a node.
  std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>>
  splitNode(Node &node) const {
    if (node.isLeaf()) {
      return r_tree::impl::splitNode<T, Object>(
          node, minElements, [](Node &node) -> llvm::SmallVectorImpl<Object> & {
            return node.objects;
          });
    }

    auto result = r_tree::impl::splitNode<T, std::unique_ptr<Node>>(
        node, minElements,
        [](Node &node) -> llvm::SmallVectorImpl<std::unique_ptr<Node>> & {
          return node.children;
        });

    // The old children node have been moved to a different node, so we need to
    // update their parent.
    for (auto &child : result.first->children) {
      child->setParent(result.first.get());
    }

    for (auto &child : result.second->children) {
      child->setParent(result.second.get());
    }

    return result;
  }

  /// Collapse a node and its ancestors, if needed. The remaining objects and
  /// subtrees are relocated.
  void collapseNodeAndPropagate(Node *node) {
    if (node->isRoot()) {
      return;
    }

    // Additional nodes to be collapsed may be obtained during the process.
    llvm::SmallVector<Node *> nodesToCollapse;
    nodesToCollapse.push_back(node);

    while (!nodesToCollapse.empty()) {
      node = nodesToCollapse.pop_back_val();

      // Move the objects out of the node being removed.
      // This is done only once, as the nodes that will be possibly visited
      // later will not be leaves.
      llvm::SmallVector<Object> objectsToRelocate = std::move(node->objects);

      while (!node->isRoot() && node->fanOut() < minElements) {
        // The nodes to be relocated.
        llvm::SmallVector<std::unique_ptr<Node>> nodesToRelocate;

        // Move the objects out of the node being removed.
        for (auto &child : node->children) {
          nodesToRelocate.push_back(std::move(child));
        }

        // Remove the node from the list of children of the parent.
        // The node itself will be erased, and any usage of 'node' will be
        // invalid. Therefore, we need to store the reference to the parent
        // elsewhere.
        Node *parent = node->getParent();
        removeChild(parent, node);

        // Relocate the orphaned objects.
        if (!objectsToRelocate.empty() && !parent->children.empty()) {
          for (Object &object : objectsToRelocate) {
            // Find the new position.
            Node *destination = chooseLeaf(parent, object.getBoundary());

            // Add the value to the leaf node.
            destination->add(std::move(object));
            postObjectInsertionHook(*destination);

            if (destination->fanOut() > maxElements) {
              // Split the node.
              splitNodeAndPropagate(destination);
            } else if (destination->fanOut() < minElements) {
              // Schedule the node for removal.
              nodesToCollapse.push_back(destination);
            } else {
              updateBoundaries(destination);
            }
          }

          objectsToRelocate.clear();
        }

        // Relocate the orphaned nodes.
        for (auto &nodeToRelocate : nodesToRelocate) {
          // Find the new position.
          Node *destination = chooseDestinationNode(
              parent->children, nodeToRelocate->getBoundary());

          // Add the value to the leaf node.
          destination->add(std::move(nodeToRelocate));

          if (destination->fanOut() > maxElements) {
            // Split the node.
            splitNodeAndPropagate(destination);
          } else {
            updateBoundaries(destination);
          }
        }

        // Move towards the root of the tree.
        node = parent;
      }

      // Update the boundaries.
      updateBoundaries(node);

      assert(objectsToRelocate.empty() &&
             "Not all objects have been relocated");

      if (getRoot() && getRoot()->children.size() == 1) {
        setRoot(std::move(getRoot()->children.front()));
      }
    }
  }

  /// Remove a child for a node.
  static void removeChild(Node *parent, const Node *child) {
    llvm::SmallVector<std::unique_ptr<Node>> newChildren;

    for (auto &currentChild : parent->children) {
      if (currentChild.get() != child) {
        newChildren.push_back(std::move(currentChild));
      }
    }

    parent->children = std::move(newChildren);
  }

  /// @name Hooks.
  /// {

  virtual void postObjectInsertionHook(Node &node) {
    // Do nothing.
  }

  virtual void postObjectRemovalHook(Node &node) {
    // Do nothing.
  }

  /// }
  /// @name Walk methods.
  /// {

private:
  template <typename NodeT>
  static void walkOverlappingNodes(NodeT *root,
                                   const MultidimensionalRange &range,
                                   llvm::function_ref<void(NodeT &)> walkFn) {
    llvm::SmallVector<NodeT *> nodeStack;

    if (root) {
      nodeStack.push_back(root);
    }

    while (!nodeStack.empty()) {
      NodeT *node = nodeStack.pop_back_val();

      if (node->getBoundary().overlaps(range)) {
        walkFn(*node);
      }

      for (auto &child :
           llvm::make_range(node->children.rbegin(), node->children.rend())) {
        nodeStack.push_back(child.get());
      }
    }
  }

protected:
  void walkOverlappingNodes(const MultidimensionalRange &range,
                            llvm::function_ref<void(Node &)> walkFn) {
    walkOverlappingNodes<Node>(getRoot(), range, walkFn);
  }

  void
  walkOverlappingNodes(const MultidimensionalRange &range,
                       llvm::function_ref<void(const Node &)> walkFn) const {
    walkOverlappingNodes<Node>(getRoot(), range, walkFn);
  }

private:
  template <typename NodeT>
  static void
  walkOverlappingLeafNodes(NodeT *root, const MultidimensionalRange &range,
                           llvm::function_ref<void(NodeT &)> walkFn) {
    walkOverlappingNodes<NodeT>(root, range, [&](NodeT &node) {
      if (node.isLeaf()) {
        walkFn(node);
      }
    });
  }

protected:
  void walkOverlappingLeafNodes(const MultidimensionalRange &range,
                                llvm::function_ref<void(Node &)> walkFn) {
    walkOverlappingLeafNodes(getRoot(), range, walkFn);
  }

  void walkOverlappingLeafNodes(
      const MultidimensionalRange &range,
      llvm::function_ref<void(const Node &)> walkFn) const {
    walkOverlappingLeafNodes(getRoot(), range, walkFn);
  }

private:
  template <typename NodeT, typename ValueT>
  static void
  walkOverlappingObjects(NodeT *root, const MultidimensionalRange &range,
                         llvm::function_ref<void(ValueT &)> walkFn) {
    walkOverlappingLeafNodes<NodeT>(root, range, [&](NodeT &node) {
      for (auto &object : node.objects) {
        if (object.getBoundary().overlaps(range)) {
          walkFn(*object);
        }
      }
    });
  }

public:
  void walkOverlappingObjects(const MultidimensionalRange &range,
                              llvm::function_ref<void(T &)> walkFn) {
    walkOverlappingObjects(getRoot(), range, walkFn);
  }

  void
  walkOverlappingObjects(const MultidimensionalRange &range,
                         llvm::function_ref<void(const T &)> walkFn) const {
    walkOverlappingObjects(getRoot(), range, walkFn);
  }

  /// }

  /// Check if all the invariants are respected.
  bool isValid() const {
#ifndef NDEBUG
    if (!checkParentRelationships(*this)) {
      return false;
    }

    if (!checkMBRsInvariant(*this)) {
      return false;
    }

    if (!checkFanOutInvariant(*this)) {
      return false;
    }
#endif

    return true;
  }

#ifndef NDEBUG
  /// Check that all the children of a node has the correct parent set.
  static bool checkParentRelationships(const RTreeCRTP &rTree) {
    llvm::SmallVector<const Node *> nodes;

    if (auto root = rTree.getRoot()) {
      nodes.push_back(root);
    }

    while (!nodes.empty()) {
      auto node = nodes.pop_back_val();

      for (const auto &child : node->children) {
        if (child->getParent() != node) {
          return false;
        }

        nodes.push_back(child.get());
      }
    }

    return true;
  }

  /// Check the correctness of the MBR of all the nodes.
  static bool checkMBRsInvariant(const RTreeCRTP &rTree) {
    llvm::SmallVector<const Node *> nodes;

    if (auto root = rTree.getRoot()) {
      nodes.push_back(root);
    }

    while (!nodes.empty()) {
      auto node = nodes.pop_back_val();

      if (node->isLeaf()) {
        if (node->getBoundary() != getMBR(llvm::ArrayRef(node->objects))) {
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
  static bool checkFanOutInvariant(const RTreeCRTP &rTree) {
    llvm::SmallVector<const Node *> nodes;

    if (const Node *node = rTree.getRoot()) {
      for (const auto &child : node->children) {
        nodes.push_back(child.get());
      }
    }

    while (!nodes.empty()) {
      auto node = nodes.pop_back_val();

      if (size_t size = node->fanOut();
          size < rTree.minElements || size > rTree.maxElements) {
        return false;
      }

      for (const auto &child : node->children) {
        nodes.push_back(child.get());
      }
    }

    return true;
  }
#endif
};
} // namespace r_tree::impl

template <typename T>
class RTree : public r_tree::impl::RTreeCRTP<RTree<T>, T> {
public:
  auto begin() const { return this->objectsBegin(); }
  auto end() const { return this->objectsEnd(); }
};
} // namespace marco::modeling

#endif // MARCO_MODELING_RTREE_H
