#ifndef MARCO_MODELING_GRAPH_H
#define MARCO_MODELING_GRAPH_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include <functional>
#include <memory>

namespace marco::modeling::internal {
namespace impl {
/// Light-weight class referring to a vertex of the graph.
template <typename Graph, typename P>
struct VertexDescriptor {
public:
  using Property = P;

  VertexDescriptor() : graph(nullptr), value(nullptr) {}

  VertexDescriptor(const Graph *graph, Property *value, uint64_t id)
      : graph(std::move(graph)), value(std::move(value)), id(id) {}

  friend llvm::hash_code hash_value(const VertexDescriptor &descriptor) {
    return llvm::hash_value(descriptor.id);
  }

  bool operator==(const VertexDescriptor &other) const {
    return graph == other.graph && id == other.id;
  }

  bool operator!=(const VertexDescriptor &other) const {
    return graph != other.graph || id != other.id;
  }

  bool operator<(const VertexDescriptor &other) const { return id < other.id; }

  const Graph *graph;
  Property *value;
  uint64_t id{0};
};

/// Light-weight class referring to an edge of the graph.
template <typename Graph, typename T, typename VertexDescriptor>
struct EdgeDescriptor {
public:
  EdgeDescriptor() : graph(nullptr), value(nullptr) {}

  EdgeDescriptor(const Graph *graph, VertexDescriptor from, VertexDescriptor to,
                 T *value)
      : graph(std::move(graph)), from(std::move(from)), to(std::move(to)),
        value(std::move(value)) {}

  friend llvm::hash_code hash_value(const EdgeDescriptor &descriptor) {
    return llvm::hash_combine(descriptor.from, descriptor.to);
  }

  bool operator==(const EdgeDescriptor &other) const {
    return from == other.from && to == other.to && value == other.value;
  }

  bool operator!=(const EdgeDescriptor &other) const {
    return from != other.from || to != other.to || value != other.value;
  }

  bool operator<(const EdgeDescriptor &other) const {
    if (from != to) {
      return from < to;
    }

    return false;
  }

  const Graph *graph;
  VertexDescriptor from;
  VertexDescriptor to;
  T *value;
};

template <typename T>
class PropertyWrapper {
public:
  PropertyWrapper(T property) : value(std::move(property)) {}

  friend llvm::hash_code hash_value(const PropertyWrapper &val) {
    return hash_value(val.value);
  }

  T &operator*() { return value; }

  const T &operator*() const { return value; }

private:
  T value;
};

template <typename VertexProperty, typename EdgeProperty>
class VertexWrapper;

template <typename VertexProperty, typename EdgeProperty>
class EdgeWrapper;

template <typename VertexProperty, typename EdgeProperty>
class VertexWrapper
    : public PropertyWrapper<VertexProperty>,
      public llvm::DGNode<VertexWrapper<VertexProperty, EdgeProperty>,
                          EdgeWrapper<VertexProperty, EdgeProperty>> {
  using EW = EdgeWrapper<VertexProperty, EdgeProperty>;

  uint64_t id;
  llvm::MapVector<VertexWrapper *, llvm::SetVector<EW *>> incomingEdges;

public:
  explicit VertexWrapper(VertexProperty property, uint64_t id)
      : PropertyWrapper<VertexProperty>(std::move(property)), id(id) {}

  uint64_t getId() const { return id; }

  auto getIncomingEdges() {
    return llvm::make_range(incomingEdges.begin(), incomingEdges.end());
  }

  auto getIncomingEdges() const {
    return llvm::make_range(incomingEdges.begin(), incomingEdges.end());
  }

  void addIncomingEdge(VertexWrapper *from, EW *edge) {
    assert(edge->getTargetNode() == *this);
    incomingEdges[from].insert(edge);
  }

  void removeIncomingEdge(VertexWrapper *from, EW *edge) {
    assert(incomingEdges[from].contains(edge));
    incomingEdges[from].remove(edge);
  }
};

template <typename VertexProperty, typename EdgeProperty>
class EdgeWrapper
    : public PropertyWrapper<EdgeProperty>,
      public llvm::DGEdge<VertexWrapper<VertexProperty, EdgeProperty>,
                          EdgeWrapper<VertexProperty, EdgeProperty>> {
public:
  EdgeWrapper(VertexWrapper<VertexProperty, EdgeProperty> &destination,
              EdgeProperty property)
      : PropertyWrapper<EdgeProperty>(std::move(property)),
        llvm::DGEdge<VertexWrapper<VertexProperty, EdgeProperty>,
                     EdgeWrapper<VertexProperty, EdgeProperty>>(destination) {}
};

/// Utility class to group the common properties of a graph.
template <typename Derived, typename VP, typename EP>
struct GraphTraits {
  using Type = Derived;

  using VertexProperty = VP;
  using EdgeProperty = EP;

  // We use the LLVM's directed graph implementation.
  // However, nodes and edges are not owned by the LLVM's graph
  // implementation, and thus we need to manage their lifetime.
  // Moreover, in case of undirected graph the edge property should be
  // shared among the two directed edges connecting two nodes. This is why
  // the edge property is used as pointer.
  using Vertex = VertexWrapper<VertexProperty, EdgeProperty *>;
  using Edge = EdgeWrapper<VertexProperty, EdgeProperty *>;

  class Base : public llvm::DirectedGraph<Vertex, Edge> {
  public:
    /// Add a vertex that is known to be unique.
    bool addUniqueNode(Vertex &node) {
      // Override base implementation, as it scales quadratically due to the
      // search for duplicates.
      this->Nodes.push_back(&node);
      return true;
    }

    /// Remove a vertex from the list of vertices.
    void removeNodeOnly(Vertex &node) {
      auto it = llvm::find_if(this->Nodes, [&node](const Vertex *current) {
        return *current == node;
      });

      if (it != this->Nodes.end()) {
        this->Nodes.erase(it);
      }
    }
  };

  using BaseVertexIterator = typename Base::const_iterator;
  using BaseEdgeIterator = typename llvm::DGNode<Vertex, Edge>::const_iterator;

  using VertexDescriptor = impl::VertexDescriptor<Derived, Vertex>;
  using EdgeDescriptor = impl::EdgeDescriptor<Derived, Edge, VertexDescriptor>;
};

template <typename GraphTraits>
class VertexIterator {
private:
  using VertexDescriptor = typename GraphTraits::VertexDescriptor;
  using Graph = typename GraphTraits::Type;
  using Vertex = typename GraphTraits::Vertex;
  using BaseGraph = typename GraphTraits::Base;
  using BaseVertexIterator = typename GraphTraits::BaseVertexIterator;

public:
  using iterator_category = std::input_iterator_tag;
  using value_type = VertexDescriptor;
  using difference_type = std::ptrdiff_t;
  using pointer = VertexDescriptor *;

  // Input iterator does not require the 'reference' type to be an actual
  // reference.
  using reference = VertexDescriptor;

private:
  VertexIterator(const Graph &graph, BaseVertexIterator current)
      : graph(&graph), current(std::move(current)) {}

public:
  bool operator==(const VertexIterator &it) const {
    return current == it.current;
  }

  bool operator!=(const VertexIterator &it) const {
    return current != it.current;
  }

  VertexIterator &operator++() {
    ++current;
    return *this;
  }

  VertexIterator operator++(int) {
    auto temp = *this;
    ++current;
    return temp;
  }

  VertexDescriptor operator*() const {
    Vertex *vertex = *current;
    return VertexDescriptor(graph, vertex, vertex->getId());
  }

  /// @name Construction methods
  /// {

  static VertexIterator begin(const Graph &graph, const BaseGraph &baseGraph) {
    return VertexIterator(graph, baseGraph.begin());
  }

  static VertexIterator end(const Graph &graph, const BaseGraph &baseGraph) {
    return VertexIterator(graph, baseGraph.end());
  }

  /// }

private:
  const Graph *graph;
  BaseVertexIterator current;
};

template <typename GraphTraits>
class EdgeIterator {
private:
  using VertexDescriptor = typename GraphTraits::VertexDescriptor;
  using EdgeDescriptor = typename GraphTraits::EdgeDescriptor;
  using Graph = typename GraphTraits::Type;
  using Vertex = typename GraphTraits::Vertex;
  using BaseGraph = typename GraphTraits::Base;
  using BaseVertexIterator = typename GraphTraits::BaseVertexIterator;
  using BaseEdgeIterator = typename GraphTraits::BaseEdgeIterator;

public:
  using iterator_category = std::input_iterator_tag;
  using value_type = EdgeDescriptor;
  using difference_type = std::ptrdiff_t;
  using pointer = EdgeDescriptor *;

  // Input iterator does not require the 'reference' type to be an actual
  // reference.
  using reference = EdgeDescriptor;

private:
  EdgeIterator(const Graph &graph, bool directed,
               BaseVertexIterator currentVertexIt,
               BaseVertexIterator endVertexIt,
               std::optional<BaseEdgeIterator> currentEdgeIt,
               std::optional<BaseEdgeIterator> endEdgeIt)
      : graph(&graph), directed(directed),
        currentVertexIt(std::move(currentVertexIt)),
        endVertexIt(std::move(endVertexIt)),
        currentEdgeIt(std::move(currentEdgeIt)),
        endEdgeIt(std::move(endEdgeIt)) {
    fetchNext();
  }

public:
  bool operator==(const EdgeIterator &it) const {
    return currentVertexIt == it.currentVertexIt &&
           currentEdgeIt == it.currentEdgeIt;
  }

  bool operator!=(const EdgeIterator &it) const {
    return currentVertexIt != it.currentVertexIt ||
           currentEdgeIt != it.currentEdgeIt;
  }

  EdgeIterator &operator++() {
    advance();
    return *this;
  }

  EdgeIterator operator++(int) {
    auto temp = *this;
    advance();
    return temp;
  }

  EdgeDescriptor operator*() const {
    Vertex *sourceNode = *currentVertexIt;
    VertexDescriptor source(graph, sourceNode, sourceNode->getId());
    Vertex &targetNode = (**currentEdgeIt)->getTargetNode();
    VertexDescriptor destination(graph, &targetNode, targetNode.getId());

    return EdgeDescriptor(graph, std::move(source), std::move(destination),
                          **currentEdgeIt);
  }

  /// @name Construction methods
  /// {

  static EdgeIterator begin(const Graph &graph, bool directed,
                            const BaseGraph &baseGraph) {
    auto currentVertexIt = baseGraph.begin();
    auto endVertexIt = baseGraph.end();

    if (currentVertexIt == endVertexIt) {
      // There are no vertices. The current vertex iterator is already
      // past-the-end and thus we must avoid dereferencing it.
      return EdgeIterator(graph, directed, currentVertexIt, endVertexIt,
                          std::nullopt, std::nullopt);
    }

    auto currentEdgeIt = (*currentVertexIt)->begin();
    auto endEdgeIt = (*currentVertexIt)->end();

    return EdgeIterator(graph, directed, currentVertexIt, endVertexIt,
                        currentEdgeIt, endEdgeIt);
  }

  static EdgeIterator end(const Graph &graph, bool directed,
                          const BaseGraph &baseGraph) {
    return EdgeIterator(graph, directed, baseGraph.end(), baseGraph.end(),
                        std::nullopt, std::nullopt);
  }

  /// }

private:
  bool shouldProceed() const {
    if (currentVertexIt == endVertexIt) {
      return false;
    }

    if (currentEdgeIt == endEdgeIt) {
      return true;
    }

    if (directed) {
      return false;
    }

    Vertex *source = *currentVertexIt;
    Vertex *destination = &(**currentEdgeIt)->getTargetNode();

    return source->getId() < destination->getId();
  }

  void fetchNext() {
    while (shouldProceed()) {
      bool advanceToNextVertex = currentEdgeIt == endEdgeIt;

      if (advanceToNextVertex) {
        ++currentVertexIt;

        if (currentVertexIt == endVertexIt) {
          currentEdgeIt = std::nullopt;
          endEdgeIt = std::nullopt;
        } else {
          currentEdgeIt = (*currentVertexIt)->begin();
          endEdgeIt = (*currentVertexIt)->end();
        }
      } else {
        ++(*currentEdgeIt);
      }
    }
  }

  void advance() {
    ++(*currentEdgeIt);
    fetchNext();
  }

private:
  const Graph *graph;
  bool directed;
  BaseVertexIterator currentVertexIt;
  BaseVertexIterator endVertexIt;
  std::optional<BaseEdgeIterator> currentEdgeIt;
  std::optional<BaseEdgeIterator> endEdgeIt;
};

template <typename GraphTraits>
class IncidentEdgeIterator {
private:
  using VertexDescriptor = typename GraphTraits::VertexDescriptor;
  using EdgeDescriptor = typename GraphTraits::EdgeDescriptor;
  using Graph = typename GraphTraits::Type;
  using Vertex = typename GraphTraits::Vertex;
  using BaseGraph = typename GraphTraits::Base;
  using BaseEdgeIterator = typename GraphTraits::BaseEdgeIterator;

public:
  using iterator_category = std::input_iterator_tag;
  using value_type = EdgeDescriptor;
  using difference_type = std::ptrdiff_t;
  using pointer = EdgeDescriptor *;

  // Input iterator does not require the 'reference' type to be an actual
  // reference.
  using reference = EdgeDescriptor;

private:
  IncidentEdgeIterator(const Graph &graph, VertexDescriptor from,
                       BaseEdgeIterator current)
      : graph(&graph), from(std::move(from)), current(std::move(current)) {}

public:
  bool operator==(const IncidentEdgeIterator &it) const {
    return from == it.from && current == it.current;
  }

  bool operator!=(const IncidentEdgeIterator &it) const {
    return from != it.from || current != it.current;
  }

  IncidentEdgeIterator &operator++() {
    ++current;
    return *this;
  }

  IncidentEdgeIterator operator++(int) {
    auto temp = *this;
    ++current;
    return temp;
  }

  EdgeDescriptor operator*() const {
    auto &edge = *current;
    auto &targetNode = edge->getTargetNode();
    VertexDescriptor to(graph, &targetNode, targetNode.getId());
    return EdgeDescriptor(graph, from, to, *current);
  }

  /// @name Construction methods
  /// {

  static IncidentEdgeIterator begin(const Graph &graph,
                                    const BaseGraph &baseGraph,
                                    const Vertex &source,
                                    VertexDescriptor sourceDescriptor) {
    return IncidentEdgeIterator(graph, sourceDescriptor, source.begin());
  }

  static IncidentEdgeIterator end(const Graph &graph,
                                  const BaseGraph &baseGraph,
                                  const Vertex &source,
                                  VertexDescriptor sourceDescriptor) {
    return IncidentEdgeIterator(graph, sourceDescriptor, source.end());
  }

  /// }

private:
  const Graph *graph;
  VertexDescriptor from;
  BaseEdgeIterator current;
};

template <typename GraphTraits>
class LinkedVerticesIterator {
private:
  using VertexDescriptor = typename GraphTraits::VertexDescriptor;
  using Graph = typename GraphTraits::Type;
  using Vertex = typename GraphTraits::Vertex;
  using BaseGraph = typename GraphTraits::Base;
  using BaseEdgeIterator = typename GraphTraits::BaseEdgeIterator;

public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = VertexDescriptor;
  using difference_type = std::ptrdiff_t;
  using pointer = VertexDescriptor *;
  using reference = VertexDescriptor &;

private:
  LinkedVerticesIterator(const Graph &graph, BaseEdgeIterator current)
      : graph(&graph), current(std::move(current)) {}

public:
  bool operator==(const LinkedVerticesIterator &it) const {
    return current == it.current;
  }

  bool operator!=(const LinkedVerticesIterator &it) const {
    return current != it.current;
  }

  LinkedVerticesIterator &operator++() {
    ++current;
    return *this;
  }

  LinkedVerticesIterator operator++(int) {
    auto temp = *this;
    ++current;
    return temp;
  }

  VertexDescriptor operator*() const {
    auto &edge = *current;
    auto &targetNode = edge->getTargetNode();
    return VertexDescriptor(graph, &targetNode, targetNode.getId());
  }

  /// @name Construction methods
  /// {

  static LinkedVerticesIterator
  begin(const Graph &graph, const BaseGraph &baseGraph, const Vertex &from) {
    return LinkedVerticesIterator(graph, from.begin());
  }

  static LinkedVerticesIterator
  end(const Graph &graph, const BaseGraph &baseGraph, const Vertex &from) {
    return LinkedVerticesIterator(graph, from.end());
  }

  /// }

private:
  const Graph *graph;
  BaseEdgeIterator current;
};
} // namespace impl

template <typename Derived, typename VP, typename EP>
class Graph {
private:
  using Traits = impl::GraphTraits<Graph<Derived, VP, EP>, VP, EP>;
  using Vertex = typename Traits::Vertex;
  using Edge = typename Traits::Edge;

public:
  using VertexProperty = typename Traits::VertexProperty;
  using EdgeProperty = typename Traits::EdgeProperty;

  using VertexDescriptor = typename Traits::VertexDescriptor;
  using EdgeDescriptor = typename Traits::EdgeDescriptor;

  using VertexIterator = impl::VertexIterator<Traits>;

  using FilteredVertexIterator =
      llvm::filter_iterator<VertexIterator,
                            std::function<bool(VertexDescriptor)>>;

  using EdgeIterator = impl::EdgeIterator<Traits>;

  using FilteredEdgeIterator =
      llvm::filter_iterator<EdgeIterator, std::function<bool(EdgeDescriptor)>>;

  using IncidentEdgeIterator = impl::IncidentEdgeIterator<Traits>;

  using FilteredIncidentEdgeIterator =
      llvm::filter_iterator<IncidentEdgeIterator,
                            std::function<bool(EdgeDescriptor)>>;

  using LinkedVerticesIterator = impl::LinkedVerticesIterator<Traits>;

  using FilteredLinkedVerticesIterator =
      llvm::filter_iterator<LinkedVerticesIterator,
                            std::function<bool(VertexDescriptor)>>;

  Graph(bool directed) : directed(std::move(directed)) {}

  // Cloning the descriptors would lead to double deallocations when both
  // the original and the new graph would get destructed.
  Graph(const Graph &other) = delete;

  // Moving the graph would result in an invalidation of vertex descriptors,
  // which contain the address of the graph.
  Graph(Graph &&other) = delete;

  virtual ~Graph() {
    for (EdgeProperty *edgeProperty : edgeProperties) {
      delete edgeProperty;
    }

    for (Edge *edge : edges) {
      delete edge;
    }

    for (Vertex *vertex : vertices) {
      delete vertex;
    }
  }

  Graph &operator=(const Graph &other) = delete;

  Graph &operator=(Graph &&other) = delete;

  /// @name Data access methods
  /// {

  /// Get the vertex property given its descriptor.
  VertexProperty &operator[](VertexDescriptor descriptor) {
    return getVertexPropertyFromDescriptor(descriptor);
  }

  /// Get the vertex property given its descriptor.
  const VertexProperty &operator[](VertexDescriptor descriptor) const {
    return getVertexPropertyFromDescriptor(descriptor);
  }

  /// Get the edge property given its descriptor.
  EdgeProperty &operator[](EdgeDescriptor descriptor) {
    return getEdgePropertyFromDescriptor(descriptor);
  }

  /// Get the edge property given its descriptor.
  const EdgeProperty &operator[](EdgeDescriptor descriptor) const {
    return getEdgePropertyFromDescriptor(descriptor);
  }

  /// }

  /// Get the number of vertices.
  size_t verticesCount() const { return vertices.size(); }

  /// Get the number of edges.
  size_t edgesCount() const {
    if (directed) {
      return edges.size();
    }

    return edges.size() / 2;
  }

  /// Add a vertex to the graph.
  /// The property is cloned and its lifetime is tied to the graph.
  VertexDescriptor addVertex(VertexProperty property) {
    uint64_t id = nextVertexId++;
    auto *ptr = new Vertex(std::move(property), id);
    vertices.insert(ptr);
    [[maybe_unused]] bool vertexInsertion = graph.addUniqueNode(*ptr);
    assert(vertexInsertion);
    return VertexDescriptor(this, ptr, id);
  }

  /// Remove a vertex from the graph.
  void removeVertex(VertexDescriptor vertexDescriptor) {
    Vertex &vertex = getVertexFromDescriptor(vertexDescriptor);

    // Remove the incoming edges.
    for (auto &incomingEdges : vertex.getIncomingEdges()) {
      // Remove the edge from the outgoing set of edges of the other vertex.
      Vertex &otherVertex = *incomingEdges.first;

      for (Edge *incomingEdge : incomingEdges.second) {
        otherVertex.removeEdge(*incomingEdge);

        if (directed) {
          // Deallocate the edge property only in case of directed graph. In
          // case of undirected graph, the edge property will be deallocated
          // while visiting the outgoing edges (in an undirected graph the edge
          // property is shared between the wrapper, and a double deallocation
          // would otherwise occur).
          edgeProperties.remove(**incomingEdge);
          delete **incomingEdge;
        }

        // Deallocate the edge wrapper.
        edges.remove(incomingEdge);
        delete incomingEdge;
      }
    }

    // Remove the outgoing edges.
    for (Edge *edge : vertex) {
      // Remove the edge from the incoming set of edges of the other vertex.
      Vertex &otherVertex = edge->getTargetNode();
      otherVertex.removeIncomingEdge(&vertex, edge);

      // Deallocate the edge property.
      edgeProperties.remove(**edge);
      delete **edge;

      // Deallocate the edge wrapper.
      edges.remove(edge);
      delete edge;
    }

    // Remove the vertex.
    graph.removeNodeOnly(vertex);
    vertices.remove(vertexDescriptor.value);
    delete vertexDescriptor.value;
  }

  /// Get the vertices of the graph.
  auto getVertices() const {
    return llvm::make_range(verticesBegin(), verticesEnd());
  }

  /// Get the begin iterator for the vertices of the graph.
  VertexIterator verticesBegin() const {
    return VertexIterator::begin(*this, graph);
  }

  /// Get the end iterator for the vertices of the graph.
  VertexIterator verticesEnd() const {
    return VertexIterator::end(*this, graph);
  }

  /// Get the begin iterator for the vertices of the graph that
  /// match a certain property.
  ///
  /// @param visibilityFn  function determining whether a vertex should be
  /// considered or not
  /// @return vertices iterator
  FilteredVertexIterator verticesBegin(
      std::function<bool(const VertexProperty &)> visibilityFn) const {
    auto filter = [this, visibilityFn](VertexDescriptor descriptor) -> bool {
      return visibilityFn(this->getVertexPropertyFromDescriptor(descriptor));
    };

    return FilteredVertexIterator(verticesBegin(), verticesEnd(), filter);
  }

  /// Get the end iterator for the vertices of the graph that
  /// match a certain property.
  ///
  /// @param visibilityFn  function determining whether a vertex should be
  /// considered or not
  /// @return vertices iterator
  FilteredVertexIterator
  verticesEnd(std::function<bool(const VertexProperty &)> visibilityFn) const {
    auto filter = [this, visibilityFn](VertexDescriptor descriptor) -> bool {
      return visibilityFn(this->getVertexPropertyFromDescriptor(descriptor));
    };

    return FilteredVertexIterator(verticesEnd(), verticesEnd(), filter);
  }

  /// Add an edge to the graph.
  /// The property is cloned and stored into the graph data structures.
  EdgeDescriptor addEdge(VertexDescriptor from, VertexDescriptor to,
                         EdgeProperty property = EdgeProperty()) {
    Vertex &src = getVertexFromDescriptor(from);
    Vertex &dest = getVertexFromDescriptor(to);

    // Allocate the property on the heap, so that it can be shared between
    // both the edges, in case of undirected graph.
    auto *edgeProperty = new EdgeProperty(std::move(property));
    edgeProperties.insert(edgeProperty);
    auto *ptr = new Edge(dest, edgeProperty);
    edges.insert(ptr);
    [[maybe_unused]] bool edgeInsertion = graph.connect(src, dest, *ptr);
    assert(edgeInsertion);
    dest.addIncomingEdge(&src, ptr);

    if (!directed) {
      // If the graph is undirected, add also the edge going from the
      // destination to the source.
      auto *inversePtr = new Edge(src, edgeProperty);
      edges.insert(inversePtr);

      [[maybe_unused]] bool inverseEdgeInsertion =
          graph.connect(dest, src, *inversePtr);

      assert(inverseEdgeInsertion);
      src.addIncomingEdge(&dest, inversePtr);
    }

    return EdgeDescriptor(this, from, to, ptr);
  }

  /// Get the edges of the graph.
  /// If the graph is undirected, then the edge between two nodes is
  /// returned only once and its source / destination order is casual,
  /// as it is indeed conceptually irrelevant.
  auto getEdges() const { return llvm::make_range(edgesBegin(), edgesEnd()); }

  /// Get the begin iterator for all the edges of the graph.
  /// If the graph is undirected, then the edge between two nodes is
  /// returned only once and its source / destination order is casual,
  /// as it is indeed conceptually irrelevant.
  EdgeIterator edgesBegin() const {
    return EdgeIterator::begin(*this, directed, graph);
  }

  /// Get the end iterator for all the edges of the graph.
  /// If the graph is undirected, then the edge between two nodes is
  /// returned only once and its source / destination order is casual,
  /// as it is indeed conceptually irrelevant.
  EdgeIterator edgesEnd() const {
    return EdgeIterator::end(*this, directed, graph);
  }

  /// Get the begin iterator for all the edges of the graph that match
  /// a certain property. If the graph is undirected, then the edge
  /// between two nodes is returned only once and its source / destination
  /// order is casual, as it is indeed conceptually irrelevant.
  ///
  /// @param visibilityFn  function determining whether an edge should be
  /// considered or not
  /// @return iterator
  FilteredEdgeIterator
  edgesBegin(std::function<bool(const EdgeProperty &)> visibilityFn) const {
    auto filter = [this, visibilityFn](EdgeDescriptor descriptor) -> bool {
      return visibilityFn(this->getEdgePropertyFromDescriptor(descriptor));
    };

    return FilteredEdgeIterator(edgesBegin(), edgesEnd(), filter);
  }

  /// Get the end iterator for all the edges of the graph that match
  /// a certain property. If the graph is undirected, then the edge
  /// between two nodes is returned only once and its source / destination
  /// order is casual, as it is indeed conceptually irrelevant.
  ///
  /// @param visibilityFn  function determining whether an edge should be
  /// considered or not
  /// @return iterator
  FilteredEdgeIterator
  edgesEnd(std::function<bool(const EdgeProperty &)> visibilityFn) const {
    auto filter = [this, visibilityFn](EdgeDescriptor descriptor) -> bool {
      return visibilityFn(this->getEdgePropertyFromDescriptor(descriptor));
    };

    return FilteredEdgeIterator(edgesEnd(), edgesEnd(), filter);
  }

  /// Get the edges exiting from a node.
  auto getOutgoingEdges(VertexDescriptor vertex) const {
    return llvm::make_range(outgoingEdgesBegin(vertex),
                            outgoingEdgesEnd(vertex));
  }

  /// Get the begin iterator for the edges exiting from a node.
  IncidentEdgeIterator outgoingEdgesBegin(VertexDescriptor vertex) const {
    const Vertex &v = getVertexFromDescriptor(vertex);
    return IncidentEdgeIterator::begin(*this, graph, v, vertex);
  }

  /// Get the end iterator for the edges exiting from a node.
  IncidentEdgeIterator outgoingEdgesEnd(VertexDescriptor vertex) const {
    const Vertex &v = getVertexFromDescriptor(vertex);
    return IncidentEdgeIterator::end(*this, graph, v, vertex);
  }

  /// Get the edges exiting from a node that match a certain property.
  auto
  getOutgoingEdges(VertexDescriptor vertex,
                   std::function<bool(const EdgeProperty &)> visibilityFn) {
    return llvm::make_range(outgoingEdgesBegin(vertex, visibilityFn),
                            outgoingEdgesEnd(vertex, visibilityFn));
  }

  /// Get the begin iterator for the edges exiting from a node that
  /// match a certain property.
  ///
  /// @param vertex        source vertex
  /// @param visibilityFn  function determining whether an edge
  ///                      should be considered or not
  /// @return iterator
  FilteredIncidentEdgeIterator outgoingEdgesBegin(
      VertexDescriptor vertex,
      std::function<bool(const EdgeProperty &)> visibilityFn) const {
    auto filter = [this, visibilityFn](EdgeDescriptor descriptor) -> bool {
      return visibilityFn(this->getEdgePropertyFromDescriptor(descriptor));
    };

    return FilteredIncidentEdgeIterator(outgoingEdgesBegin(vertex),
                                        outgoingEdgesEnd(vertex), filter);
  }

  /// Get the end iterator for the edges exiting from a node that
  /// match a certain property.
  ///
  /// @param vertex        source vertex
  /// @param visibilityFn  function determining whether an edge
  ///                      should be considered or not
  /// @return iterator
  FilteredIncidentEdgeIterator outgoingEdgesEnd(
      VertexDescriptor vertex,
      std::function<bool(const EdgeProperty &)> visibilityFn) const {
    auto filter = [this, visibilityFn](EdgeDescriptor descriptor) -> bool {
      return visibilityFn(this->getEdgePropertyFromDescriptor(descriptor));
    };

    return FilteredIncidentEdgeIterator(outgoingEdgesEnd(vertex),
                                        outgoingEdgesEnd(vertex), filter);
  }

  /// Get the vertices connected to a given vertex.
  auto getLinkedVertices(VertexDescriptor vertex) const {
    return llvm::make_range(linkedVerticesBegin(vertex),
                            linkedVerticesEnd(vertex));
  }

  /// Get the begin iterator for the vertices connected to a vertex.
  LinkedVerticesIterator linkedVerticesBegin(VertexDescriptor vertex) const {
    const Vertex &v = getVertexFromDescriptor(vertex);
    return LinkedVerticesIterator::begin(*this, graph, v);
  }

  /// Get the end iterator for the vertices connected to a node.
  LinkedVerticesIterator linkedVerticesEnd(VertexDescriptor vertex) const {
    const Vertex &v = getVertexFromDescriptor(vertex);
    return LinkedVerticesIterator::end(*this, graph, v);
  }

  /// Get the vertices connected to a given vertex that match a certain
  /// property.
  auto getLinkedVertices(
      VertexDescriptor vertex,
      std::function<bool(const VertexProperty &)> visibilityFn) const {
    return llvm::make_range(linkedVerticesBegin(vertex, visibilityFn),
                            linkedVerticesEnd(vertex, visibilityFn));
  }

  /// Get the begin iterator for the vertices connected to a vertex
  /// that match a certain property.
  ///
  /// @param vertex        source vertex
  /// @param visibilityFn  function determining whether a vertex
  ///                      should be considered or not
  /// @return iterator
  FilteredLinkedVerticesIterator linkedVerticesBegin(
      VertexDescriptor vertex,
      std::function<bool(const VertexProperty &)> visibilityFn) const {
    auto filter = [this, visibilityFn](VertexDescriptor descriptor) -> bool {
      return visibilityFn(this->getVertexPropertyFromDescriptor(descriptor));
    };

    return FilteredLinkedVerticesIterator(linkedVerticesBegin(vertex),
                                          linkedVerticesEnd(vertex), filter);
  }

  /// Get the end iterator for the vertices connected to a node
  /// that match a certain property.
  ///
  /// @param vertex        source vertex
  /// @param visibilityFn  function determining whether a vertex
  ///                      should be considered or not
  /// @return iterator
  FilteredLinkedVerticesIterator linkedVerticesEnd(
      VertexDescriptor vertex,
      std::function<bool(const VertexProperty &)> visibilityFn) const {
    auto filter = [this, visibilityFn](VertexDescriptor descriptor) -> bool {
      return visibilityFn(this->getVertexPropertyFromDescriptor(descriptor));
    };

    return FilteredLinkedVerticesIterator(linkedVerticesEnd(vertex),
                                          linkedVerticesEnd(vertex), filter);
  }

  /// Split the graph into multiple independent ones, if possible.
  std::vector<Derived> getDisjointSubGraphs() const {
    std::vector<Derived> result;

    llvm::DenseSet<VertexDescriptor> visited;
    llvm::DenseMap<VertexDescriptor, VertexDescriptor> newVertices;

    for (VertexDescriptor vertex :
         llvm::make_range(verticesBegin(), verticesEnd())) {
      if (visited.contains(vertex)) {
        // If the node has already been visited, then it already belongs
        // to an identified sub-graph. The same holds for its connected
        // nodes.
        continue;
      }

      // Instead, if the node has not been visited yet, then a new
      // connected component is found. Thus create a new graph to hold
      // the connected nodes.

      visited.insert(vertex);
      auto &subGraph = result.emplace_back();
      newVertices.try_emplace(
          vertex, subGraph.addVertex(getVertexPropertyFromDescriptor(vertex)));

      // Depth-first search
      llvm::SmallVector<VertexDescriptor> stack;
      stack.push_back(vertex);

      while (!stack.empty()) {
        VertexDescriptor currentVertex = stack.pop_back_val();
        VertexDescriptor mappedCurrentVertex =
            newVertices.find(currentVertex)->second;

        for (EdgeDescriptor edgeDescriptor : getOutgoingEdges(currentVertex)) {
          const VertexDescriptor &child = edgeDescriptor.to;

          if (visited.contains(child)) {
            const VertexDescriptor &mappedChild =
                newVertices.find(child)->second;
            subGraph.addEdge(mappedCurrentVertex, mappedChild,
                             getEdgePropertyFromDescriptor(edgeDescriptor));
          } else {
            stack.push_back(child);
            visited.insert(child);
            VertexDescriptor mappedChild =
                subGraph.addVertex(getVertexPropertyFromDescriptor(child));
            newVertices.try_emplace(child, mappedChild);
            subGraph.addEdge(mappedCurrentVertex, mappedChild,
                             getEdgePropertyFromDescriptor(edgeDescriptor));
          }
        }
      }
    }

    return result;
  }

private:
  /// Get a vertex as it is stored in the base graph.
  Vertex &getVertexFromDescriptor(VertexDescriptor descriptor) {
    return *descriptor.value;
  }

  /// Get a vertex as it is stored in the base graph.
  const Vertex &getVertexFromDescriptor(VertexDescriptor descriptor) const {
    return *descriptor.value;
  }

  /// Get the vertex property of a vertex.
  VertexProperty &unwrapVertex(Vertex &vertex) { return *vertex; }

  /// Get the vertex property of a vertex.
  const VertexProperty &unwrapVertex(const Vertex &vertex) const {
    return *vertex;
  }

  /// Get the vertex property given its descriptor.
  VertexProperty &getVertexPropertyFromDescriptor(VertexDescriptor descriptor) {
    Vertex &vertex = getVertexFromDescriptor(descriptor);
    return unwrapVertex(vertex);
  }

  /// Get the vertex property given its descriptor.
  const VertexProperty &
  getVertexPropertyFromDescriptor(VertexDescriptor descriptor) const {
    const Vertex &vertex = getVertexFromDescriptor(descriptor);
    return unwrapVertex(vertex);
  }

  /// Get an edge as it is stored in the base graph.
  Edge &getEdgeFromDescriptor(EdgeDescriptor descriptor) {
    return *descriptor.value;
  }

  /// Get an edge as it is stored in the base graph.
  const Edge &getEdgeFromDescriptor(EdgeDescriptor descriptor) const {
    return *descriptor.value;
  }

  /// Get the edge property given its descriptor.
  EdgeProperty &getEdgePropertyFromDescriptor(EdgeDescriptor descriptor) {
    Edge &edge = getEdgeFromDescriptor(descriptor);
    return **edge;
  }

  /// Get the edge property given its descriptor.
  const EdgeProperty &
  getEdgePropertyFromDescriptor(EdgeDescriptor descriptor) const {
    const Edge &edge = getEdgeFromDescriptor(descriptor);
    return **edge;
  }

private:
  bool directed;
  llvm::SetVector<Vertex *> vertices;
  uint64_t nextVertexId{0};
  llvm::SetVector<Edge *> edges;
  llvm::SetVector<EdgeProperty *> edgeProperties;
  typename Traits::Base graph;
};

/// Default edge property.
class EmptyEdgeProperty {};

/// Undirected graph.
template <typename VP, typename EP = internal::EmptyEdgeProperty>
class UndirectedGraph : public Graph<UndirectedGraph<VP, EP>, VP, EP> {
private:
  using Base = internal::Graph<UndirectedGraph<VP, EP>, VP, EP>;

public:
  using VertexProperty = typename Base::VertexProperty;
  using EdgeProperty = typename Base::EdgeProperty;

  using VertexDescriptor = typename Base::VertexDescriptor;
  using EdgeDescriptor = typename Base::EdgeDescriptor;

  using VertexIterator = typename Base::VertexIterator;
  using FilteredVertexIterator = typename Base::FilteredVertexIterator;

  using EdgeIterator = typename Base::EdgeIterator;
  using FilteredEdgeIterator = typename Base::FilteredEdgeIterator;

  using IncidentEdgeIterator = typename Base::IncidentEdgeIterator;
  using FilteredIncidentEdgeIterator =
      typename Base::FilteredIncidentEdgeIterator;

  UndirectedGraph() : Base(false) {}

  UndirectedGraph(const UndirectedGraph &other) = delete;

  UndirectedGraph(UndirectedGraph &&other) = delete;

  ~UndirectedGraph() override = default;

  UndirectedGraph &operator=(const UndirectedGraph &other) = delete;

  UndirectedGraph &operator=(UndirectedGraph &&other) = delete;
};

/// Directed graph.
template <typename VP, typename EP = internal::EmptyEdgeProperty>
class DirectedGraph : public Graph<DirectedGraph<VP, EP>, VP, EP> {
private:
  using Base = internal::Graph<DirectedGraph<VP, EP>, VP, EP>;

public:
  using VertexProperty = typename Base::VertexProperty;
  using EdgeProperty = typename Base::EdgeProperty;

  using VertexDescriptor = typename Base::VertexDescriptor;
  using EdgeDescriptor = typename Base::EdgeDescriptor;

  using VertexIterator = typename Base::VertexIterator;
  using FilteredVertexIterator = typename Base::FilteredVertexIterator;

  using EdgeIterator = typename Base::EdgeIterator;
  using FilteredEdgeIterator = typename Base::FilteredEdgeIterator;

  using IncidentEdgeIterator = typename Base::IncidentEdgeIterator;
  using FilteredIncidentEdgeIterator =
      typename Base::FilteredIncidentEdgeIterator;

  DirectedGraph() : Base(true) {}

  DirectedGraph(const DirectedGraph &other) = delete;

  DirectedGraph(DirectedGraph &&other) = delete;

  ~DirectedGraph() override = default;

  DirectedGraph &operator=(const DirectedGraph &other) = delete;

  DirectedGraph &operator=(DirectedGraph &&other) = delete;
};

template <typename T>
struct UniquePtrWrapper {
  std::unique_ptr<T> value;

  UniquePtrWrapper(std::unique_ptr<T> value) : value(std::move(value)) {}

  UniquePtrWrapper(T *value) : value(value) {}

  friend llvm::hash_code hash_value(const UniquePtrWrapper &val) {
    return hash_value(*val.value);
  }

  operator std::unique_ptr<T> &() { return value; }

  operator const std::unique_ptr<T> &() { return value; }

  template <typename U>
  bool operator==(const U &other) const {
    return value == other;
  }

  template <typename U>
  bool operator==(U &&other) const {
    return value == other;
  }

  template <typename U>
  bool operator!=(const U &other) const {
    return value != other;
  }

  template <typename U>
  bool operator!=(U &&other) const {
    return value != other;
  }

  T &operator*() {
    assert(value && "Null pointer");
    return *value;
  }

  const T &operator*() const {
    assert(value && "Null pointer");
    return *value;
  }

  T *operator->() {
    assert(value && "Null pointer");
    return value.get();
  }

  const T *operator->() const {
    assert(value && "Null pointer");
    return value.get();
  }
};
} // namespace marco::modeling::internal

namespace llvm {
template <typename Graph, typename VertexProperty>
struct DenseMapInfo<
    marco::modeling::internal::impl::VertexDescriptor<Graph, VertexProperty>> {
  using Key =
      marco::modeling::internal::impl::VertexDescriptor<Graph, VertexProperty>;

  static Key getEmptyKey() {
    return Key(llvm::DenseMapInfo<const Graph *>::getEmptyKey(),
               llvm::DenseMapInfo<VertexProperty *>::getEmptyKey(),
               llvm::DenseMapInfo<uint64_t>::getEmptyKey());
  }

  static Key getTombstoneKey() {
    return Key(llvm::DenseMapInfo<const Graph *>::getTombstoneKey(),
               llvm::DenseMapInfo<VertexProperty *>::getTombstoneKey(),
               llvm::DenseMapInfo<uint64_t>::getTombstoneKey());
  }

  static unsigned getHashValue(const Key &Val) { return hash_value(Val); }

  static bool isEqual(const Key &LHS, const Key &RHS) {
    return LHS.value == RHS.value;
  }
};

template <typename Graph, typename T, typename VertexDescriptor>
struct DenseMapInfo<marco::modeling::internal::impl::EdgeDescriptor<
    Graph, T, VertexDescriptor>> {
  using Key = marco::modeling::internal::impl::EdgeDescriptor<Graph, T,
                                                              VertexDescriptor>;

  static Key getEmptyKey() {
    auto emptyVertex = llvm::DenseMapInfo<VertexDescriptor>::getEmptyKey();
    return Key(nullptr, emptyVertex, emptyVertex, nullptr);
  }

  static Key getTombstoneKey() {
    auto tombstoneVertex =
        llvm::DenseMapInfo<VertexDescriptor>::getTombstoneKey();

    return Key(nullptr, tombstoneVertex, tombstoneVertex, nullptr);
  }

  static unsigned getHashValue(const Key &Val) { return hash_value(Val); }

  static bool isEqual(const Key &LHS, const Key &RHS) {
    return LHS.value == RHS.value;
  }
};
} // namespace llvm

#endif // MARCO_MODELING_GRAPH_H
