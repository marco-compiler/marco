#ifndef MARCO_MODELING_SINGLEENTRYDIGRAPH_H
#define MARCO_MODELING_SINGLEENTRYDIGRAPH_H

#include "marco/Modeling/Graph.h"
#include "llvm/ADT/GraphTraits.h"

namespace marco::modeling::dependency {
template <typename VP, typename EP = internal::EmptyEdgeProperty>
class SingleEntryDigraph {
public:
  using VertexProperty = VP;
  using EdgeProperty = EP;

private:
  using Graph = internal::DirectedGraph<std::unique_ptr<VertexProperty>,
                                        std::unique_ptr<EdgeProperty>>;

public:
  using VertexDescriptor = typename Graph::VertexDescriptor;
  using EdgeDescriptor = typename Graph::EdgeDescriptor;

  using VertexIterator = typename Graph::VertexIterator;
  using IncidentEdgeIterator = typename Graph::IncidentEdgeIterator;
  using LinkedVerticesIterator = typename Graph::LinkedVerticesIterator;

  SingleEntryDigraph() : entryNode(graph.addVertex(nullptr)) {}

  VertexProperty &operator[](VertexDescriptor vertex) {
    assert(vertex != entryNode && "The entry node doesn't have a property");
    return *graph[vertex];
  }

  const VertexProperty &operator[](VertexDescriptor vertex) const {
    assert(vertex != entryNode && "The entry node doesn't have a property");
    return *graph[vertex];
  }

  EdgeProperty &operator[](EdgeDescriptor edge) { return *graph[edge]; }

  const EdgeProperty &operator[](EdgeDescriptor edge) const {
    return *graph[edge];
  }

  size_t size() const { return graph.verticesCount(); }

  VertexDescriptor getEntryNode() const { return entryNode; }

  VertexDescriptor addVertex(VertexProperty property) {
    auto descriptor =
        graph.addVertex(std::make_unique<VertexProperty>(std::move(property)));

    return descriptor;
  }

  auto verticesBegin() const {
    return graph.verticesBegin(
        [](const typename Graph::VertexProperty &vertex) {
          // Hide the entry point
          return vertex != nullptr;
        });
  }

  auto verticesEnd() const {
    return graph.verticesEnd([](const typename Graph::VertexProperty &vertex) {
      // Hide the entry point
      return vertex != nullptr;
    });
  }

  EdgeDescriptor addEdge(VertexDescriptor from, VertexDescriptor to,
                         EdgeProperty property = EdgeProperty()) {
    return graph.addEdge(from, to,
                         std::make_unique<EdgeProperty>(std::move(property)));
  }

  auto getEdges() const { return graph.getEdges(); }

  auto outgoingEdgesBegin(VertexDescriptor vertex) const {
    return graph.outgoingEdgesBegin(std::move(vertex));
  }

  auto outgoingEdgesEnd(VertexDescriptor vertex) const {
    return graph.outgoingEdgesEnd(std::move(vertex));
  }

  auto linkedVerticesBegin(VertexDescriptor vertex) const {
    return graph.linkedVerticesBegin(std::move(vertex));
  }

  auto linkedVerticesEnd(VertexDescriptor vertex) const {
    return graph.linkedVerticesEnd(std::move(vertex));
  }

private:
  Graph graph;
  VertexDescriptor entryNode;
};
} // namespace marco::modeling::dependency

namespace llvm {
// We specialize the LLVM's graph traits in order leverage the algorithms
// that are defined inside LLVM itself. This way we don't have to implement
// them from scratch.
template <typename VertexProperty, typename EdgeProperty>
struct GraphTraits<const marco::modeling::dependency ::SingleEntryDigraph<
    VertexProperty, EdgeProperty> *> {
  // The LLVM traits require the class specified as Graph to be copyable.
  // We use its address to overcome this limitation.
  using Graph =
      const marco::modeling::dependency::SingleEntryDigraph<VertexProperty>;

  using GraphPtr = Graph *;

  using NodeRef = typename Graph::VertexDescriptor;
  using ChildIteratorType = typename Graph::LinkedVerticesIterator;

  static NodeRef getEntryNode(const GraphPtr &graph) {
    return graph->getEntryNode();
  }

  static ChildIteratorType child_begin(NodeRef node) {
    return node.graph->linkedVerticesBegin(node);
  }

  static ChildIteratorType child_end(NodeRef node) {
    return node.graph->linkedVerticesEnd(node);
  }

  using nodes_iterator = typename Graph::VertexIterator;

  static nodes_iterator nodes_begin(GraphPtr *graph) {
    return (*graph)->verticesBegin();
  }

  static nodes_iterator nodes_end(GraphPtr *graph) {
    return (*graph)->verticesEnd();
  }

  using EdgeRef = typename Graph::EdgeDescriptor;
  using ChildEdgeIteratorType = typename Graph::IncidentEdgeIterator;

  static ChildEdgeIteratorType child_edge_begin(NodeRef node) {
    return node.graph->outgoingEdgesBegin(node);
  }

  static ChildEdgeIteratorType child_edge_end(NodeRef node) {
    return node.graph->outgoingEdgesEnd(node);
  }

  static NodeRef edge_dest(EdgeRef edge) { return edge.to; }

  static size_t size(GraphPtr *graph) { return (*graph)->size(); }
};
} // namespace llvm

#endif // MARCO_MODELING_SINGLEENTRYDIGRAPH_H
