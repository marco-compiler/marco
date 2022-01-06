#ifndef MARCO_MODELING_GRAPH_H
#define MARCO_MODELING_GRAPH_H

#include <functional>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/ADT/SmallVector.h>
#include <map>
#include <stack>

namespace marco::modeling::internal
{
  namespace impl
  {
    template<typename Graph, typename P>
    struct VertexDescriptor
    {
      public:
        using Property = P;

        VertexDescriptor(const Graph* graph, Property* value)
            : graph(std::move(graph)), value(std::move(value))
        {
        }

        bool operator==(const VertexDescriptor& other) const
        {
          return value == other.value;
        }

        bool operator!=(const VertexDescriptor& other) const
        {
          return value != other.value;
        }

        bool operator<(const VertexDescriptor& other) const
        {
          return value < other.value;
        }

        const Graph* graph;
        Property* value;
    };

    template<typename Graph, typename T, typename VertexDescriptor>
    struct EdgeDescriptor
    {
      public:
        EdgeDescriptor(const Graph* graph, VertexDescriptor from, VertexDescriptor to, T* value)
            : graph(std::move(graph)), from(std::move(from)), to(std::move(to)), value(std::move(value))
        {
        }

        bool operator==(const EdgeDescriptor<Graph, T, VertexDescriptor>& other) const
        {
          return from == other.from && to == other.to && value == other.value;
        }

        bool operator!=(const EdgeDescriptor<Graph, T, VertexDescriptor>& other) const
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
          if (shouldProceed()) {
            fetchNext();
          }
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
          if (currentIt == endIt) {
            return false;
          }

          auto descriptor = *currentIt;
          return !visibilityFn(descriptor);
        }

        void fetchNext()
        {
          if (currentIt == endIt) {
            return;
          }

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

        EdgeIterator(const Graph* graph, VertexIterator currentVertexIt, VertexIterator endVertexIt,
                     const AdjacencyList& adj, bool directed)
            : graph(std::move(graph)),
              currentVertexIt(std::move(currentVertexIt)),
              endVertexIt(std::move(endVertexIt)),
              adj(&adj),
              currentEdge(0),
              directed(directed)
        {
          if (shouldProceed()) {
            fetchNext();
          }
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
          if (currentVertexIt == endVertexIt) {
            return false;
          }

          VertexDescriptor from = *currentVertexIt;
          auto incidentEdgesIt = adj->find(from);

          if (incidentEdgesIt == adj->end()) {
            return true;
          }

          auto& incidentEdges = incidentEdgesIt->second;

          if (currentEdge == incidentEdges.size()) {
            return true;
          }

          auto& edge = incidentEdges[currentEdge];

          if (directed) {
            return false;
          }

          return from < edge.first;
        }

        void fetchNext()
        {
          if (currentVertexIt == endVertexIt) {
            return;
          }

          do {
            VertexDescriptor from = *currentVertexIt;

            auto incidentEdgesIt = adj->find(from);
            bool advanceToNextVertex = incidentEdgesIt == adj->end();

            if (!advanceToNextVertex) {
              auto& incidentEdges = incidentEdgesIt->second;
              advanceToNextVertex = currentEdge == incidentEdges.size();
            }

            if (advanceToNextVertex) {
              ++currentVertexIt;
              currentEdge = 0;
            } else {
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
  }

  template<template<typename, typename> class Derived, typename VP, typename EP>
  class Graph
  {
    public:
      using VertexProperty = VP;
      using EdgeProperty = EP;

      using VertexDescriptor = impl::VertexDescriptor<
          Graph<Derived, VertexProperty, EdgeProperty>,
          VertexProperty>;

      using EdgeDescriptor = impl::EdgeDescriptor<
          Graph<Derived, VertexProperty, EdgeProperty>,
          EdgeProperty,
          VertexDescriptor>;

    private:
      using VerticesContainer = llvm::SmallVector<VertexProperty*, 3>;
      using IncidentEdgesList = std::vector<std::pair<VertexDescriptor, EdgeProperty*>>;
      using AdjacencyList = std::map<VertexDescriptor, IncidentEdgesList>;

    public:
      using VertexIterator = impl::VertexIterator<
          Graph<Derived, VertexProperty, EdgeProperty>,
          VertexDescriptor,
          VerticesContainer>;

      using FilteredVertexIterator = impl::FilteredIterator<VertexIterator>;

      using EdgeIterator = impl::EdgeIterator<
          Graph<Derived, VertexProperty, EdgeProperty>,
          EdgeDescriptor,
          VertexIterator,
          AdjacencyList>;

      using FilteredEdgeIterator = impl::FilteredIterator<EdgeIterator>;

      using IncidentEdgeIterator = impl::IncidentEdgeIterator<
          Graph<Derived, VertexProperty, EdgeProperty>,
          VertexDescriptor,
          EdgeDescriptor,
          IncidentEdgesList>;

      using FilteredIncidentEdgeIterator = impl::FilteredIterator<IncidentEdgeIterator>;

      using LinkedVerticesIterator = impl::LinkedVerticesIterator<
          Graph<Derived, VertexProperty, EdgeProperty>,
          VertexDescriptor,
          EdgeDescriptor,
          IncidentEdgesList>;

      using FilteredLinkedVerticesIterator = impl::FilteredIterator<LinkedVerticesIterator>;

      Graph(bool directed) : directed(std::move(directed))
      {
      }

      Graph(const Graph& other) : directed(other.directed)
      {
        // Perform a deep copy of the graph.
        // Cloning the descriptors is in fact insufficient, as the wrapped nodes
        // would be deallocated once the original graph ceases to exist.

        llvm::DenseMap<VertexDescriptor, VertexDescriptor> verticesMap;
        llvm::DenseMap<VertexDescriptor, VertexDescriptor> edgesMap;

        for (auto vertex: other.getVertices()) {
          auto vertexClone = addVertex(this->operator[](vertex));
          verticesMap.try_emplace(vertex, vertexClone);
        }

        for (auto edge: other.getEdges()) {
          auto from = verticesMap.find(edge.from)->second;
          auto to = verticesMap.find(edge.to)->second;
          addEdge(from, to, this->operator[](edge));
        }
      }

      virtual ~Graph()
      {
        for (auto* vertex: vertices) {
          delete vertex;
        }

        for (auto* edge: edges) {
          delete edge;
        }
      }

      Graph& operator=(const Graph& other)
      {
        Graph result(other);
        swap(*this, result);
        return *this;
      }

      friend void swap(Graph& first, Graph& second)
      {
        using std::swap;
        swap(first.directed, second.directed);
        swap(first.vertices, second.vertices);
        swap(first.edges, second.edges);
        swap(first.adj, second.adj);
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

      /**
       * Get the number of vertices.
       *
       * @return vertices count
       */
      size_t verticesCount() const
      {
        return vertices.size();
      }

      /**
       * Get the number of edges.
       *
       * @return edges count
       */
      size_t edgesCount() const
      {
        if (directed) {
          return edges.size();
        }

        return edges.size() / 2;
      }

      /**
       * Add a vertex to the graph.
       * The property is cloned and stored into the graph data structures.
       *
       * @param property  vertex property
       * @return vertex descriptor
       */
      VertexDescriptor addVertex(VertexProperty property)
      {
        auto* ptr = new VertexProperty(std::move(property));
        vertices.push_back(ptr);
        VertexDescriptor result(this, ptr);
        // TODO Replace with multimap
        adj.emplace(result, IncidentEdgesList());
        return result;
      }

      /**
       * Get the vertices of the graph.
       *
       * @return iterable range of vertex descriptors
       */
      llvm::iterator_range<VertexIterator> getVertices() const
      {
        VertexIterator begin(this, vertices.begin());
        VertexIterator end(this, vertices.end());

        return llvm::iterator_range<VertexIterator>(begin, end);
      }

      /**
       * Get the vertices of the graph that match a certain property.
       *
       * @param visibilityFn  function determining whether a vertex should be considered or not
       * @return iterable range of vertex descriptors
       */
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

      /**
       * Add an edge to the graph.
       * The property is cloned and stored into the graph data structures.
       *
       * @param from        source vertex descriptor
       * @param to          destination vertex descriptor
       * @param property    edge property
       * @return edge descriptor
       */
      EdgeDescriptor addEdge(
          VertexDescriptor from,
          VertexDescriptor to,
          EdgeProperty property = EdgeProperty())
      {
        auto* ptr = new EdgeProperty(std::move(property));
        edges.push_back(ptr);

        adj[from].push_back(std::make_pair(to, ptr));

        if (!directed) {
          adj[to].push_back(std::make_pair(from, ptr));
        }

        return EdgeDescriptor(this, from, to, ptr);
      }

      /**
       * Get all the edges of the graph.
       * If the graph is undirected, then the edge between two nodes is
       * returned only once and its source / destination order is casual,
       * as it is indeed conceptually irrelevant.
       *
       * @return iterable range of edge descriptors
       */
      llvm::iterator_range<EdgeIterator> getEdges() const
      {
        auto verticesDescriptors = this->getVertices();

        EdgeIterator begin(this, verticesDescriptors.begin(), verticesDescriptors.end(), adj, directed);
        EdgeIterator end(this, verticesDescriptors.end(), verticesDescriptors.end(), adj, directed);

        return llvm::iterator_range<EdgeIterator>(begin, end);
      }

      /**
       * Get all the edges of the graph that match a certain property.
       * If the graph is undirected, then the edge between two nodes is
       * returned only once and its source / destination order is casual,
       * as it is indeed conceptually irrelevant.
       *
       * @param visibilityFn  function determining whether an edge should be considered or not
       * @return iterable range of edge descriptors
       */
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

      llvm::iterator_range<IncidentEdgeIterator> getOutgoingEdges(VertexDescriptor vertex) const
      {
        auto it = adj.find(vertex);
        assert(it != this->adj.end());
        const auto& incidentEdges = it->second;

        IncidentEdgeIterator begin(this, vertex, incidentEdges.begin());
        IncidentEdgeIterator end(this, vertex, incidentEdges.end());

        return llvm::iterator_range<IncidentEdgeIterator>(begin, end);
      }

      llvm::iterator_range<FilteredIncidentEdgeIterator> getOutgoingEdges(
          VertexDescriptor vertex,
          std::function<bool(const EdgeProperty&)> visibilityFn) const
      {
        auto filter = [=](const EdgeDescriptor& descriptor) -> bool {
          return visibilityFn((*this)[descriptor]);
        };

        auto allEdges = getOutgoingEdges(vertex);

        FilteredIncidentEdgeIterator begin(allEdges.begin(), allEdges.end(), filter);
        FilteredIncidentEdgeIterator end(allEdges.end(), allEdges.end(), filter);

        return llvm::iterator_range<FilteredIncidentEdgeIterator>(begin, end);
      }

      llvm::iterator_range<LinkedVerticesIterator> getLinkedVertices(VertexDescriptor vertex) const
      {
        auto incidentEdges = getOutgoingEdges(vertex);

        LinkedVerticesIterator begin(incidentEdges.begin());
        LinkedVerticesIterator end(incidentEdges.end());

        return llvm::iterator_range<LinkedVerticesIterator>(begin, end);
      }

      llvm::iterator_range<FilteredLinkedVerticesIterator> getOutgoingEdges(
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
       * Split the graph into multiple independent ones, if possible.
       *
       * @return disjoint graphs
       */
      std::vector<Derived<VertexProperty, EdgeProperty>> getDisjointSubGraphs() const
      {
        std::vector<Derived<VertexProperty, EdgeProperty>> result;

        llvm::DenseSet<VertexDescriptor> visited;
        llvm::DenseMap<VertexDescriptor, VertexDescriptor> newVertices;

        for (auto vertex: getVertices()) {
          if (visited.contains(vertex)) {
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

          while (!stack.empty()) {
            auto currentVertex = stack.top();
            stack.pop();

            auto mappedCurrentVertex = newVertices.find(currentVertex)->second;

            for (auto edgeDescriptor: getOutgoingEdges(currentVertex)) {
              auto child = edgeDescriptor.to;

              if (visited.contains(child)) {
                auto mappedChild = newVertices.find(child)->second;
                subGraph.addEdge(mappedCurrentVertex, mappedChild, this->operator[](edgeDescriptor));
              } else {
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

  /**
   * Default edge property.
   */
  class EmptyEdgeProperty
  {
  };

  template<typename VP, typename EP = internal::EmptyEdgeProperty>
  class UndirectedGraph : public Graph<UndirectedGraph, VP, EP>
  {
    private:
      using Base = internal::Graph<UndirectedGraph, VP, EP>;

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
      using FilteredIncidentEdgeIterator = typename Base::FilteredIncidentEdgeIterator;

      UndirectedGraph() : Base(false)
      {
      }

      UndirectedGraph(const UndirectedGraph& other) = default;

      UndirectedGraph(UndirectedGraph&& other) = default;

      ~UndirectedGraph() = default;

      UndirectedGraph& operator=(const UndirectedGraph& other) = default;
  };

  template<typename VP, typename EP = internal::EmptyEdgeProperty>
  class DirectedGraph : public Graph<DirectedGraph, VP, EP>
  {
    private:
      using Base = internal::Graph<DirectedGraph, VP, EP>;

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
      using FilteredIncidentEdgeIterator = typename Base::FilteredIncidentEdgeIterator;

      DirectedGraph() : Base(true)
      {
      }

      DirectedGraph(const DirectedGraph& other) = default;

      DirectedGraph(DirectedGraph&& other) = default;

      ~DirectedGraph() = default;

      DirectedGraph& operator=(const DirectedGraph& other) = default;
  };
}

namespace llvm
{
  template<typename Graph, typename AccessProperty>
  struct DenseMapInfo<marco::modeling::internal::impl::VertexDescriptor<Graph, AccessProperty>>
  {
    using Key = marco::modeling::internal::impl::VertexDescriptor<Graph, AccessProperty>;

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
}

#endif  // MARCO_MODELING_GRAPH_H
