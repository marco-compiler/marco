#ifndef MARCO_MATCHING_GRAPH_H
#define MARCO_MATCHING_GRAPH_H

#include <functional>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/ADT/SmallVector.h>
#include <map>
#include <stack>

namespace marco::matching::detail
{
  template<typename Graph, typename P>
  struct VertexDescriptorWrapper
  {
    public:
    using Property = P;

    VertexDescriptorWrapper(const Graph* graph, Property* value)
          : graph(std::move(graph)), value(std::move(value))
    {
    }

    bool operator==(const VertexDescriptorWrapper<Graph, Property>& other) const
    {
      return value == other.value;
    }

    bool operator!=(const VertexDescriptorWrapper<Graph, Property>& other) const
    {
      return value != other.value;
    }

    bool operator<(const VertexDescriptorWrapper<Graph, Property>& other) const
    {
      return value < other.value;
    }

    const Graph* graph;
    Property* value;
  };

  template<typename Graph, typename T, typename VertexDescriptor>
  struct EdgeDescriptorWrapper
  {
    public:
    EdgeDescriptorWrapper(const Graph* graph, VertexDescriptor from, VertexDescriptor to, T* value)
            : graph(std::move(graph)), from(std::move(from)), to(std::move(to)), value(std::move(value))
    {
    }

    bool operator==(const EdgeDescriptorWrapper<Graph, T, VertexDescriptor>& other) const
    {
      return from == other.from && to == other.to && value == other.value;
    }

    bool operator!=(const EdgeDescriptorWrapper<Graph, T, VertexDescriptor>& other) const
    {
      return from != other.from || to != other.to || value != other.value;
    }

    const Graph* graph;
    VertexDescriptor from;
    VertexDescriptor to;
    T* value;
  };

  // TODO: use SFINAE to check that Iterator iterates on values with Descriptor type
  template<typename Iterator>
  class FilteredIterator
  {
    public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = typename Iterator::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = typename Iterator::pointer;
    using reference = typename Iterator::reference;

    FilteredIterator(Iterator currentIt, Iterator endIt, std::function<bool(value_type)> visibilityFn)
            : currentIt(std::move(currentIt)),
              endIt(std::move(endIt)),
              visibilityFn(std::move(visibilityFn))
    {
      if (shouldProceed())
        fetchNext();
    }

    bool operator==(const FilteredIterator& it) const
    {
      return currentIt == it.currentIt && endIt == it.endIt;
    }

    bool operator!=(const FilteredIterator& it) const
    {
      return currentIt != it.currentIt || endIt != it.endIt;
    }

    FilteredIterator& operator++()
    {
      fetchNext();
      return *this;
    }

    FilteredIterator operator++(int)
    {
      auto temp = *this;
      fetchNext();
      return temp;
    }

    value_type operator*() const
    {
      return *currentIt;
    }

    private:
    bool shouldProceed() const
    {
      if (currentIt == endIt)
        return false;

      auto descriptor = *currentIt;
      return !visibilityFn(descriptor);
    }

    void fetchNext()
    {
      if (currentIt == endIt)
        return;

      do {
        ++currentIt;
      } while (shouldProceed());
    }

    Iterator currentIt;
    Iterator endIt;
    std::function<bool(value_type)> visibilityFn;
  };

  template<typename Graph, typename VertexDescriptor, typename VerticesContainer>
  class VertexIterator
  {
    public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = VertexDescriptor;
    using difference_type = std::ptrdiff_t;
    using pointer = VertexDescriptor*;
    using reference = VertexDescriptor&;

    private:
    using Iterator = typename VerticesContainer::const_iterator;

    public:
    VertexIterator(const Graph* graph, Iterator current)
            : graph(std::move(graph)), current(std::move(current))
    {
    }

    bool operator==(const VertexIterator& it) const
    {
      return current == it.current;
    }

    bool operator!=(const VertexIterator& it) const
    {
      return current != it.current;
    }

    VertexIterator& operator++()
    {
      ++current;
      return *this;
    }

    VertexIterator operator++(int)
    {
      auto temp = *this;
      ++current;
      return temp;
    }

    VertexDescriptor operator*() const
    {
      return VertexDescriptor(graph, *current);
    }

    private:
    const Graph* graph;
    Iterator current;
  };

  template<typename Graph, typename EdgeDescriptor, typename VertexIterator, typename AdjacencyList>
  class EdgeIterator
  {
    public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = EdgeDescriptor;
    using difference_type = std::ptrdiff_t;
    using pointer = EdgeDescriptor*;
    using reference = EdgeDescriptor&;

    using VertexDescriptor = typename VertexIterator::value_type;

    EdgeIterator(const Graph* graph, VertexIterator currentVertexIt, VertexIterator endVertexIt, const AdjacencyList& adj, bool directed)
            : graph(std::move(graph)),
              currentVertexIt(std::move(currentVertexIt)),
              endVertexIt(std::move(endVertexIt)),
              adj(&adj),
              currentEdge(0),
              directed(directed)
    {
      if (shouldProceed())
        fetchNext();
    }

    bool operator==(const EdgeIterator& it) const
    {
      return currentVertexIt == it.currentVertexIt && currentEdge == it.currentEdge;
    }

    bool operator!=(const EdgeIterator& it) const
    {
      return currentVertexIt != it.currentVertexIt || currentEdge != it.currentEdge;
    }

    EdgeIterator& operator++()
    {
      fetchNext();
      return *this;
    }

    EdgeIterator operator++(int)
    {
      auto temp = *this;
      fetchNext();
      return temp;
    }

    EdgeDescriptor operator*() const
    {
      VertexDescriptor from = *currentVertexIt;
      auto& incidentEdges = adj->find(from)->second;
      auto& edge = incidentEdges[currentEdge];
      auto to = edge.first;
      auto* ptr = edge.second;
      return EdgeDescriptor(graph, from, to, ptr);
    }

    private:
    bool shouldProceed() const
    {
      if (currentVertexIt == endVertexIt)
        return false;

      VertexDescriptor from = *currentVertexIt;
      auto incidentEdgesIt = adj->find(from);

      if (incidentEdgesIt == adj->end())
        return true;

      auto& incidentEdges = incidentEdgesIt->second;

      if (currentEdge == incidentEdges.size())
        return true;

      auto& edge = incidentEdges[currentEdge];

      if (directed)
        return true;

      return from < edge.first;
    }

    void fetchNext()
    {
      if (currentVertexIt == endVertexIt)
        return;

      do
      {
        VertexDescriptor from = *currentVertexIt;

        auto incidentEdgesIt = adj->find(from);
        bool advanceToNextVertex = incidentEdgesIt == adj->end();

        if (!advanceToNextVertex)
        {
          auto& incidentEdges = incidentEdgesIt->second;
          advanceToNextVertex = currentEdge == incidentEdges.size();
        }

        if (advanceToNextVertex)
        {
          ++currentVertexIt;
          currentEdge = 0;
        }
        else
        {
          ++currentEdge;
        }
      } while (shouldProceed());
    }

    const Graph* graph;
    VertexIterator currentVertexIt;
    VertexIterator endVertexIt;
    const AdjacencyList* adj;
    size_t currentEdge;
    bool directed;
  };

  template<typename Graph, typename VertexDescriptor, typename EdgeDescriptor, typename IncidentEdgesList>
  class IncidentEdgeIterator
  {
    public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = EdgeDescriptor;
    using difference_type = std::ptrdiff_t;
    using pointer = EdgeDescriptor*;
    using reference = EdgeDescriptor&;

    using Iterator = typename IncidentEdgesList::const_iterator;

    IncidentEdgeIterator(const Graph* graph, VertexDescriptor from, Iterator current)
            : graph(std::move(graph)), from(std::move(from)), current(current)
    {
    }

    bool operator==(const IncidentEdgeIterator& it) const
    {
      return from == it.from && current == it.current;
    }

    bool operator!=(const IncidentEdgeIterator& it) const
    {
      return from != it.from || current != it.current;
    }

    IncidentEdgeIterator& operator++()
    {
      ++current;
      return *this;
    }

    IncidentEdgeIterator operator++(int)
    {
      auto temp = *this;
      ++current;
      return temp;
    }

    EdgeDescriptor operator*() const
    {
      auto& edge = *current;
      VertexDescriptor to = edge.first;
      auto* property = edge.second;
      return EdgeDescriptor(graph, from, to, property);
    }

    private:
    const Graph* graph;
    VertexDescriptor from;
    Iterator current;
  };

  template<typename Graph, typename VertexDescriptor, typename EdgeDescriptor, typename IncidentEdgesList>
  class LinkedVerticesIterator
  {
    public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = VertexDescriptor;
    using difference_type = std::ptrdiff_t;
    using pointer = VertexDescriptor*;
    using reference = VertexDescriptor&;

    using EdgeIt = IncidentEdgeIterator<Graph, VertexDescriptor, EdgeDescriptor, IncidentEdgesList>;

    LinkedVerticesIterator(EdgeIt current)
            : current(current)
    {
    }

    bool operator==(const LinkedVerticesIterator& it) const
    {
      return current == it.current;
    }

    bool operator!=(const LinkedVerticesIterator& it) const
    {
      return current != it.current;
    }

    LinkedVerticesIterator& operator++()
    {
      ++current;
      return *this;
    }

    LinkedVerticesIterator operator++(int)
    {
      auto temp = *this;
      ++current;
      return temp;
    }

    VertexDescriptor operator*() const
    {
      return (*current).to;
    }

    private:
    EdgeIt current;
  };

  template<template<typename, typename> class Derived, typename VertexProperty, typename EdgeProperty>
  class Graph
  {
    public:
    using VertexDescriptor = VertexDescriptorWrapper<
            Graph<Derived, VertexProperty, EdgeProperty>,
            VertexProperty>;

    using EdgeDescriptor = EdgeDescriptorWrapper<
            Graph<Derived, VertexProperty, EdgeProperty>,
            EdgeProperty,
            VertexDescriptor>;

    private:
    using VerticesContainer = llvm::SmallVector<VertexProperty*, 3>;
    using IncidentEdgesList = std::vector<std::pair<VertexDescriptor, EdgeProperty*>>;
    using AdjacencyList = std::map<VertexDescriptor, IncidentEdgesList>;

    public:
    using VertexIterator = detail::VertexIterator<
            Graph<Derived, VertexProperty, EdgeProperty>,
            VertexDescriptor,
            VerticesContainer>;

    using FilteredVertexIterator = detail::FilteredIterator<VertexIterator>;

    using EdgeIterator = detail::EdgeIterator<
            Graph<Derived, VertexProperty, EdgeProperty>,
            EdgeDescriptor,
            VertexIterator,
            AdjacencyList>;

    using FilteredEdgeIterator = detail::FilteredIterator<EdgeIterator>;

    using IncidentEdgeIterator = detail::IncidentEdgeIterator<
            Graph<Derived, VertexProperty, EdgeProperty>,
            VertexDescriptor,
            EdgeDescriptor,
            IncidentEdgesList>;

    using FilteredIncidentEdgeIterator = detail::FilteredIterator<IncidentEdgeIterator>;

    using LinkedVerticesIterator = detail::LinkedVerticesIterator<
            Graph<Derived, VertexProperty, EdgeProperty>,
            VertexDescriptor,
            EdgeDescriptor,
            IncidentEdgesList>;

    using FilteredLinkedVerticesIterator = detail::FilteredIterator<LinkedVerticesIterator>;

    Graph(bool directed) : directed(directed)
    {
    }

    Graph(const Graph& other) : directed(other.directed)
    {
      llvm::DenseMap<VertexDescriptor, VertexDescriptor> verticesMap;
      llvm::DenseMap<VertexDescriptor, VertexDescriptor> edgesMap;

      for (auto vertex : other.getVertices())
      {
        auto vertexClone = addVertex(this->operator[](vertex));
        verticesMap.try_emplace(vertex, vertexClone);
      }

      for (auto edge : other.getEdges())
      {
        auto from = verticesMap.find(edge.from)->second;
        auto to = verticesMap.find(edge.to)->second;
        addEdge(from, to, this->operator[](edge));
      }
    }

    virtual ~Graph()
    {
      for (auto* vertex : vertices)
        delete vertex;

      for (auto* edge : edges)
        delete edge;
    }

    VertexProperty& operator[](VertexDescriptor descriptor)
    {
      return *descriptor.value;
    }

    const VertexProperty& operator[](VertexDescriptor descriptor) const
    {
      return *descriptor.value;
    }

    virtual EdgeProperty& operator[](EdgeDescriptor descriptor)
    {
      return *descriptor.value;
    }

    virtual const EdgeProperty& operator[](EdgeDescriptor descriptor) const
    {
      return *descriptor.value;
    }

    size_t size() const
    {
      return vertices.size();
    }

    VertexDescriptor addVertex(VertexProperty property)
    {
      auto* ptr = new VertexProperty(std::move(property));
      vertices.push_back(ptr);
      VertexDescriptor result(this, ptr);
      adj.emplace(result, IncidentEdgesList());
      return result;
    }

    llvm::iterator_range<VertexIterator> getVertices() const
    {
      VertexIterator begin(this, vertices.begin());
      VertexIterator end(this, vertices.end());

      return llvm::iterator_range<VertexIterator>(begin, end);
    }

    llvm::iterator_range<FilteredVertexIterator> getVertices(
            std::function<bool(const VertexProperty&)> visibilityFn) const
    {
      auto filter = [=](const VertexDescriptor& descriptor) -> bool {
          return visibilityFn((*this)[descriptor]);
      };

      auto allVertices = getVertices();

      FilteredVertexIterator begin(allVertices.begin(), allVertices.end(), filter);
      FilteredVertexIterator end(allVertices.end(), allVertices.end(), filter);

      return llvm::iterator_range<FilteredVertexIterator>(begin, end);
    }

    EdgeDescriptor addEdge(VertexDescriptor from, VertexDescriptor to, EdgeProperty property)
    {
      auto* ptr = new EdgeProperty(std::move(property));
      edges.push_back(ptr);

      adj[from].push_back(std::make_pair(to, ptr));

      if (!directed)
        adj[to].push_back(std::make_pair(from, ptr));

      return EdgeDescriptor(this, from, to, ptr);
    }

    llvm::iterator_range<EdgeIterator> getEdges() const
    {
      auto verticesDescriptors = this->getVertices();

      EdgeIterator begin(this, verticesDescriptors.begin(), verticesDescriptors.end(), adj, directed);
      EdgeIterator end(this, verticesDescriptors.end(), verticesDescriptors.end(), adj, directed);

      return llvm::iterator_range<EdgeIterator>(begin, end);
    }

    llvm::iterator_range<FilteredEdgeIterator> getEdges(
            std::function<bool(const EdgeProperty&)> visibilityFn) const
    {
      auto filter = [=](const EdgeDescriptor& descriptor) -> bool {
          return visibilityFn((*this)[descriptor]);
      };

      auto allEdges = getEdges();

      FilteredEdgeIterator begin(allEdges.begin(), allEdges.end(), filter);
      FilteredEdgeIterator end(allEdges.end(), allEdges.end(), filter);

      return llvm::iterator_range<FilteredEdgeIterator>(begin, end);
    }

    llvm::iterator_range<IncidentEdgeIterator> getIncidentEdges(VertexDescriptor vertex) const
    {
      auto it = adj.find(vertex);
      assert(it != this->adj.end());
      const auto& incidentEdges = it->second;

      IncidentEdgeIterator begin(this, vertex, incidentEdges.begin());
      IncidentEdgeIterator end(this, vertex, incidentEdges.end());

      return llvm::iterator_range<IncidentEdgeIterator>(begin, end);
    }

    llvm::iterator_range<FilteredIncidentEdgeIterator> getIncidentEdges(
            VertexDescriptor vertex,
            std::function<bool(const EdgeProperty&)> visibilityFn) const
    {
      auto filter = [=](const EdgeDescriptor& descriptor) -> bool {
          return visibilityFn((*this)[descriptor]);
      };

      auto allEdges = getIncidentEdges(vertex);

      FilteredIncidentEdgeIterator begin(allEdges.begin(), allEdges.end(), filter);
      FilteredIncidentEdgeIterator end(allEdges.end(), allEdges.end(), filter);

      return llvm::iterator_range<FilteredIncidentEdgeIterator>(begin, end);
    }

    llvm::iterator_range<LinkedVerticesIterator> getLinkedVertices(VertexDescriptor vertex) const
    {
      auto incidentEdges = getIncidentEdges(vertex);

      LinkedVerticesIterator begin(incidentEdges.begin());
      LinkedVerticesIterator end(incidentEdges.end());

      return llvm::iterator_range<LinkedVerticesIterator>(begin, end);
    }

    llvm::iterator_range<FilteredLinkedVerticesIterator> getIncidentEdges(
            VertexDescriptor vertex,
            std::function<bool(const VertexProperty&)> visibilityFn) const
    {
      auto filter = [=](const VertexDescriptor& descriptor) -> bool {
          return visibilityFn((*this)[descriptor]);
      };

      auto allEdges = getLinkedVertices(vertex);

      FilteredLinkedVerticesIterator begin(allEdges.begin(), allEdges.end(), filter);
      FilteredLinkedVerticesIterator end(allEdges.end(), allEdges.end(), filter);

      return llvm::iterator_range<FilteredLinkedVerticesIterator>(begin, end);
    }

    /**
     * Split the graph into multiple ones, each of them containing vertices that are
     * connected among themselves.
     *
     * @return connected graphs
     */
    std::vector<Derived<VertexProperty, EdgeProperty>> getConnectedComponents() const
    {
      std::vector<Derived<VertexProperty, EdgeProperty>> result;

      llvm::DenseSet<VertexDescriptor> visited;
      llvm::DenseMap<VertexDescriptor, VertexDescriptor> newVertices;

      for (auto vertex : getVertices())
      {
        if (visited.contains(vertex))
        {
          // If the node has already been visited, then it already belongs to
          // an identified sub-graph. The same holds for its connected nodes.
          continue;
        }

        // Instead, if the node has not been visited yet, then a new connected
        // component is found. Thus create a new graph to hold the connected nodes.

        visited.insert(vertex);
        auto& subGraph = result.emplace_back();
        newVertices.try_emplace(vertex, subGraph.addVertex(this->operator[](vertex)));

        // Depth-first search
        std::stack<VertexDescriptor> stack;
        stack.push(vertex);

        while (!stack.empty())
        {
          auto currentVertex = stack.top();
          stack.pop();

          auto mappedCurrentVertex = newVertices.find(currentVertex)->second;

          for (auto edgeDescriptor : getIncidentEdges(currentVertex))
          {
            auto child = edgeDescriptor.to;

            if (visited.contains(child))
            {
              auto mappedChild = newVertices.find(child)->second;
              subGraph.addEdge(mappedCurrentVertex, mappedChild, this->operator[](edgeDescriptor));
            }
            else
            {
              stack.push(child);
              visited.insert(child);
              auto mappedChild = subGraph.addVertex(this->operator[](child));
              newVertices.try_emplace(child, mappedChild);
              subGraph.addEdge(mappedCurrentVertex, mappedChild, this->operator[](edgeDescriptor));
            }
          }
        }
      }

      return result;
    }

    private:
    bool directed;
    VerticesContainer vertices;
    llvm::SmallVector<EdgeProperty*, 3> edges;
    AdjacencyList adj;
  };

  class EmptyEdgeProperty
  {
  };

  template<typename VertexProperty, typename EdgeProperty = EmptyEdgeProperty>
  class UndirectedGraph : public Graph<UndirectedGraph, VertexProperty, EdgeProperty>
  {
    public:
    using VertexDescriptor = typename Graph<UndirectedGraph, VertexProperty, EdgeProperty>::VertexDescriptor;
    using EdgeDescriptor = typename Graph<UndirectedGraph, VertexProperty, EdgeProperty>::EdgeDescriptor;

    using VertexIterator = typename Graph<UndirectedGraph, VertexProperty, EdgeProperty>::VertexIterator;
    using FilteredVertexIterator = typename Graph<UndirectedGraph, VertexProperty, EdgeProperty>::FilteredVertexIterator;

    using EdgeIterator = typename Graph<UndirectedGraph, VertexProperty, EdgeProperty>::EdgeIterator;
    using FilteredEdgeIterator = typename Graph<UndirectedGraph, VertexProperty, EdgeProperty>::FilteredEdgeIterator;

    using IncidentEdgeIterator = typename Graph<UndirectedGraph, VertexProperty, EdgeProperty>::IncidentEdgeIterator;
    using FilteredIncidentEdgeIterator = typename Graph<UndirectedGraph, VertexProperty, EdgeProperty>::FilteredIncidentEdgeIterator;

    UndirectedGraph() : Graph<UndirectedGraph, VertexProperty, EdgeProperty>(false)
    {
    }
  };

  template<typename VertexProperty, typename EdgeProperty = EmptyEdgeProperty>
  class DirectedGraph : public Graph<DirectedGraph, VertexProperty, EdgeProperty>
  {
    public:
    using VertexDescriptor = typename Graph<DirectedGraph, VertexProperty, EdgeProperty>::VertexDescriptor;
    using EdgeDescriptor = typename Graph<DirectedGraph, VertexProperty, EdgeProperty>::EdgeDescriptor;

    using VertexIterator = typename Graph<DirectedGraph, VertexProperty, EdgeProperty>::VertexIterator;
    using FilteredVertexIterator = typename Graph<DirectedGraph, VertexProperty, EdgeProperty>::FilteredVertexIterator;

    using EdgeIterator = typename Graph<DirectedGraph, VertexProperty, EdgeProperty>::EdgeIterator;
    using FilteredEdgeIterator = typename Graph<DirectedGraph, VertexProperty, EdgeProperty>::FilteredEdgeIterator;

    using IncidentEdgeIterator = typename Graph<DirectedGraph, VertexProperty, EdgeProperty>::IncidentEdgeIterator;
    using FilteredIncidentEdgeIterator = typename Graph<DirectedGraph, VertexProperty, EdgeProperty>::FilteredIncidentEdgeIterator;

    DirectedGraph() : Graph<DirectedGraph, VertexProperty, EdgeProperty>(true)
    {
    }
  };
}

namespace llvm
{
  template<typename Graph, typename AccessProperty>
  struct DenseMapInfo<marco::matching::detail::VertexDescriptorWrapper<Graph, AccessProperty>> {
    using Key = marco::matching::detail::VertexDescriptorWrapper<Graph, AccessProperty>;

    static inline Key getEmptyKey()
    {
      return Key(nullptr, nullptr);
    }

    static inline Key getTombstoneKey()
    {
      return Key(nullptr, nullptr);
    }

    static unsigned getHashValue(const Key& Val)
    {
      return std::hash<typename Key::Property*>{}(Val.value);
    }

    static bool isEqual(const Key& LHS, const Key& RHS)
    {
      return LHS.value == RHS.value;
    }
  };

  template<typename VertexProperty, typename EdgeProperty>
  struct GraphTraits<marco::matching::detail::DirectedGraph<VertexProperty, EdgeProperty>>
  {
    using Graph = marco::matching::detail::DirectedGraph<VertexProperty, EdgeProperty>;

    using NodeRef = typename Graph::VertexDescriptor;
    using ChildIteratorType = typename Graph::LinkedVerticesIterator;

    static NodeRef getEntryNode(const Graph& graph)
    {
      return *graph.getVertices().begin();
    }

    static ChildIteratorType child_begin(NodeRef node)
    {
      auto vertices = node.graph->getLinkedVertices(node);
      return vertices.begin();
    }

    static ChildIteratorType child_end(NodeRef node)
    {
      auto vertices = node.graph->getLinkedVertices(node);
      return vertices.end();
    }

    using nodes_iterator = typename Graph::VertexIterator;

    static nodes_iterator nodes_begin(Graph* graph)
    {
      return graph->getVertices().begin();
    }

    static nodes_iterator nodes_end(Graph* graph)
    {
      return graph->getVertices().end();
    }

    using EdgeRef = typename Graph::EdgeDescriptor;
    using ChildEdgeIteratorType = typename Graph::IncidentEdgeIterator;

    static ChildEdgeIteratorType child_edge_begin(NodeRef node)
    {
      auto edges = node.graph->getIncidentEdges(node);
      return edges.begin();
    }

    static ChildEdgeIteratorType child_edge_end(NodeRef node)
    {
      auto edges = node.graph->getIncidentEdges(node);
      return edges.end();
    }

    static NodeRef edge_dest(EdgeRef edge)
    {
      return edge.to;
    }

    static size_t size(Graph* graph)
    {
      return graph->size();
    }
  };
}

#endif	// MARCO_MATCHING_GRAPH_H
