#ifndef MARCO_MODELING_GRAPH_H
#define MARCO_MODELING_GRAPH_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>
#include <map>
#include <memory>
#include <stack>

namespace marco::modeling::internal
{
  namespace impl
  {
    /// Light-weight class referring to a vertex of the graph.
    template<typename Graph, typename P>
    struct VertexDescriptor
    {
      public:
        using Property = P;

        VertexDescriptor(const Graph* graph, Property* value)
            : graph(std::move(graph)),
              value(std::move(value))
        {
        }

        bool operator==(const VertexDescriptor& other) const
        {
          return graph == other.graph && value == other.value;
        }

        bool operator!=(const VertexDescriptor& other) const
        {
          return graph != other.graph || value != other.value;
        }

        bool operator<(const VertexDescriptor& other) const
        {
          return value < other.value;
        }

        const Graph* graph;
        Property* value;
    };

    /// Light-weight class referring to an edge of the graph.
    template<typename Graph, typename T, typename VertexDescriptor>
    struct EdgeDescriptor
    {
      public:
        EdgeDescriptor(const Graph* graph, VertexDescriptor from, VertexDescriptor to, T* value)
            : graph(std::move(graph)),
              from(std::move(from)),
              to(std::move(to)),
              value(std::move(value))
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

    template<typename T>
    class PropertyWrapper
    {
      public:
        PropertyWrapper(T property) : value(std::move(property))
        {
        }

        T& operator*()
        {
          return value;
        }

        const T& operator*() const
        {
          return value;
        }

      private:
        T value;
    };

    template<typename VertexProperty, typename EdgeProperty>
    class VertexWrapper;

    template<typename VertexProperty, typename EdgeProperty>
    class EdgeWrapper;

    template<typename VertexProperty, typename EdgeProperty>
    class VertexWrapper :
        public PropertyWrapper<VertexProperty>,
        public llvm::DGNode<
            VertexWrapper<VertexProperty, EdgeProperty>,
            EdgeWrapper<VertexProperty, EdgeProperty>>
    {
      public:
        explicit VertexWrapper(VertexProperty property)
          : PropertyWrapper<VertexProperty>(std::move(property))
        {
        }
    };

    template<typename VertexProperty, typename EdgeProperty>
    class EdgeWrapper :
        public PropertyWrapper<EdgeProperty>,
        public llvm::DGEdge<
            VertexWrapper<VertexProperty, EdgeProperty>,
            EdgeWrapper<VertexProperty, EdgeProperty>>
    {
      public:
        EdgeWrapper(VertexWrapper<VertexProperty, EdgeProperty>& destination, EdgeProperty property)
          : PropertyWrapper<EdgeProperty>(std::move(property)),
            llvm::DGEdge<
                VertexWrapper<VertexProperty, EdgeProperty>,
                EdgeWrapper<VertexProperty, EdgeProperty>>(destination)
        {
        }
    };

    /// Utility class to group the common properties of a graph.
    template<typename Derived, typename VP, typename EP>
    struct GraphTraits
    {
      using Type = Derived;

      using VertexProperty = VP;
      using EdgeProperty = EP;

      // We use the LLVM's directed graph implementation.
      // However, nodes and edges are not owned by the LLVM's graph implementation
      // and thus we need to manage their lifetime.
      // Moreover, in case of undirected graph the edge property should be shared
      // among the two directed edges connecting two nodes. This is why the edge
      // property is used as pointer.
      using Vertex = VertexWrapper<VertexProperty, EdgeProperty*>;
      using Edge = EdgeWrapper<VertexProperty, EdgeProperty*>;

      using Base = llvm::DirectedGraph<Vertex, Edge>;
      using BaseVertexIterator = typename Base::const_iterator;
      using BaseEdgeIterator = typename llvm::DGNode<Vertex, Edge>::const_iterator;

      using VertexDescriptor = impl::VertexDescriptor<Derived, Vertex>;
      using EdgeDescriptor = impl::EdgeDescriptor<Derived, Edge, VertexDescriptor>;
    };

    template<typename GraphTraits>
    class VertexIterator
    {
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
        using pointer = VertexDescriptor*;

        // Input iterator does not require the 'reference' type to be an actual reference
        using reference = VertexDescriptor;

      private:
        VertexIterator(const Graph& graph, BaseVertexIterator current)
            : graph(&graph), current(std::move(current))
        {
        }

      public:
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
          Vertex* vertex = *current;
          return VertexDescriptor(graph, vertex);
        }

        /// @name Construction methods
        /// {

        static VertexIterator begin(const Graph& graph, const BaseGraph& baseGraph)
        {
          return VertexIterator(graph, baseGraph.begin());
        }

        static VertexIterator end(const Graph& graph, const BaseGraph& baseGraph)
        {
          return VertexIterator(graph, baseGraph.end());
        }

        /// }

      private:
        const Graph* graph;
        BaseVertexIterator current;
    };

    template<typename GraphTraits>
    class EdgeIterator
    {
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
        using pointer = EdgeDescriptor*;

        // Input iterator does not require the 'reference' type to be an actual reference
        using reference = EdgeDescriptor;

      private:
        EdgeIterator(const Graph& graph,
                     bool directed,
                     BaseVertexIterator currentVertexIt,
                     BaseVertexIterator endVertexIt,
                     llvm::Optional<BaseEdgeIterator> currentEdgeIt,
                     llvm::Optional<BaseEdgeIterator> endEdgeIt)
            : graph(&graph),
              directed(directed),
              currentVertexIt(std::move(currentVertexIt)),
              endVertexIt(std::move(endVertexIt)),
              currentEdgeIt(std::move(currentEdgeIt)),
              endEdgeIt(std::move(endEdgeIt))
        {
          fetchNext();
        }

      public:
        bool operator==(const EdgeIterator& it) const
        {
          return currentVertexIt == it.currentVertexIt && currentEdgeIt == it.currentEdgeIt;
        }

        bool operator!=(const EdgeIterator& it) const
        {
          return currentVertexIt != it.currentVertexIt || currentEdgeIt != it.currentEdgeIt;
        }

        EdgeIterator& operator++()
        {
          advance();
          return *this;
        }

        EdgeIterator operator++(int)
        {
          auto temp = *this;
          advance();
          return temp;
        }

        EdgeDescriptor operator*() const
        {
          VertexDescriptor source(graph, *currentVertexIt);
          VertexDescriptor destination(graph, &(**currentEdgeIt)->getTargetNode());
          return EdgeDescriptor(graph, std::move(source), std::move(destination), **currentEdgeIt);
        }

        /// @name Construction methods
        /// {

        static EdgeIterator begin(const Graph& graph, bool directed, const BaseGraph& baseGraph)
        {
          auto currentVertexIt = baseGraph.begin();
          auto endVertexIt = baseGraph.end();

          if (currentVertexIt == endVertexIt) {
            // There are no vertices. The current vertex iterator is already past-the-end and thus
            // we must avoid dereferencing it.
            return EdgeIterator(graph, directed, currentVertexIt, endVertexIt, llvm::None, llvm::None);
          }

          auto currentEdgeIt = (*currentVertexIt)->begin();
          auto endEdgeIt = (*currentVertexIt)->end();

          return EdgeIterator(graph, directed, currentVertexIt, endVertexIt, currentEdgeIt, endEdgeIt);
        }

        static EdgeIterator end(const Graph& graph, bool directed, const BaseGraph& baseGraph)
        {
          return EdgeIterator(graph, directed, baseGraph.end(), baseGraph.end(), llvm::None, llvm::None);
        }

        /// }

      private:
        bool shouldProceed() const
        {
          if (currentVertexIt == endVertexIt) {
            return false;
          }

          if (currentEdgeIt == endEdgeIt) {
            return true;
          }

          if (directed) {
            return false;
          }

          Vertex* source = *currentVertexIt;
          Vertex* destination = &(**currentEdgeIt)->getTargetNode();

          return source < destination;
        }

        void fetchNext()
        {
          while (shouldProceed()) {
            bool advanceToNextVertex = currentEdgeIt == endEdgeIt;

            if (advanceToNextVertex) {
              ++currentVertexIt;

              if (currentVertexIt == endVertexIt) {
                currentEdgeIt = llvm::None;
                endEdgeIt = llvm::None;
              } else {
                currentEdgeIt = (*currentVertexIt)->begin();
                endEdgeIt = (*currentVertexIt)->end();
              }
            } else {
              ++(*currentEdgeIt);
            }
          }
        }

        void advance()
        {
          ++(*currentEdgeIt);
          fetchNext();
        }

        const Graph* graph;
        bool directed;
        BaseVertexIterator currentVertexIt;
        BaseVertexIterator endVertexIt;
        llvm::Optional<BaseEdgeIterator> currentEdgeIt;
        llvm::Optional<BaseEdgeIterator> endEdgeIt;
    };

    template<typename GraphTraits>
    class IncidentEdgeIterator
    {
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
        using pointer = EdgeDescriptor*;

        // Input iterator does not require the 'reference' type to be an actual reference
        using reference = EdgeDescriptor;

      private:
        IncidentEdgeIterator(const Graph& graph, VertexDescriptor from, BaseEdgeIterator current)
            : graph(&graph), from(std::move(from)), current(std::move(current))
        {
        }

      public:
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
          VertexDescriptor to(graph, &edge->getTargetNode());
          return EdgeDescriptor(graph, from, to, *current);
        }

        /// @name Construction methods
        /// {

        static IncidentEdgeIterator begin(
            const Graph& graph, const BaseGraph& baseGraph, const Vertex& source, VertexDescriptor sourceDescriptor)
        {
          auto iterator = (*baseGraph.findNode(source))->begin();
          return IncidentEdgeIterator(graph, sourceDescriptor, std::move(iterator));
        }

        static IncidentEdgeIterator end(
            const Graph& graph, const BaseGraph& baseGraph, const Vertex& source, VertexDescriptor sourceDescriptor)
        {
          auto iterator = (*baseGraph.findNode(source))->end();
          return IncidentEdgeIterator(graph, sourceDescriptor, std::move(iterator));
        }

        /// }

      private:
        const Graph* graph;
        VertexDescriptor from;
        BaseEdgeIterator current;
    };

    template<typename GraphTraits>
    class LinkedVerticesIterator
    {
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
        using pointer = VertexDescriptor*;
        using reference = VertexDescriptor&;

      private:
        LinkedVerticesIterator(const Graph& graph, BaseEdgeIterator current)
          : graph(&graph), current(std::move(current))
        {
        }

      public:
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
          auto& edge = *current;
          return VertexDescriptor(graph, &edge->getTargetNode());
        }

        /// @name Construction methods
        /// {

        static LinkedVerticesIterator begin(const Graph& graph, const BaseGraph& baseGraph, const Vertex& from)
        {
          auto iterator = (*baseGraph.findNode(from))->begin();
          return LinkedVerticesIterator(graph, std::move(iterator));
        }

        static LinkedVerticesIterator end(const Graph& graph, const BaseGraph& baseGraph, const Vertex& from)
        {
          auto iterator = (*baseGraph.findNode(from))->end();
          return LinkedVerticesIterator(graph, std::move(iterator));
        }

        /// }

      private:
        const Graph* graph;
        BaseEdgeIterator current;
    };

    /// The real graph implementation.
    /// It is kept separate from the user-exposed one as the implementation also manages the deallocation
    /// of vertices and edges. The user-exposed graph is just a shared pointer to this class, so that
    /// copies of the graph object do refer to the same elements.
    template<typename Derived, typename VP, typename EP>
    class Graph
    {
      private:
        using Traits = GraphTraits<Graph<Derived, VP, EP>, VP, EP>;
        using Vertex = typename Traits::Vertex;
        using Edge = typename Traits::Edge;

      public:
        using VertexProperty = typename Traits::VertexProperty;
        using EdgeProperty = typename Traits::EdgeProperty;

        using VertexDescriptor = typename Traits::VertexDescriptor;
        using EdgeDescriptor = typename Traits::EdgeDescriptor;

        using VertexIterator = impl::VertexIterator<Traits>;

        using FilteredVertexIterator = llvm::filter_iterator<
            VertexIterator, std::function<bool(VertexDescriptor)>>;

        using EdgeIterator = impl::EdgeIterator<Traits>;

        using FilteredEdgeIterator = llvm::filter_iterator<
            EdgeIterator, std::function<bool(EdgeDescriptor)>>;

        using IncidentEdgeIterator = impl::IncidentEdgeIterator<Traits>;

        using FilteredIncidentEdgeIterator = llvm::filter_iterator<
            IncidentEdgeIterator, std::function<bool(EdgeDescriptor)>>;

        using LinkedVerticesIterator = impl::LinkedVerticesIterator<Traits>;

        using FilteredLinkedVerticesIterator = llvm::filter_iterator<
            LinkedVerticesIterator, std::function<bool(VertexDescriptor)>>;

        Graph(bool directed) : directed(std::move(directed))
        {
        }

        /// Create a deep copy of another graph.
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
          // Deallocate vertices and edges. This is safe because, thanks to the shared pointer
          // of the user-exposed class, we are sure that this method is called only when the
          // graph and all its "copies" are not needed anymore.

          for (auto* edgeProperty : edgeProperties) {
            delete edgeProperty;
          }

          for (auto* edge: edges) {
            delete edge;
          }

          for (auto* vertex: vertices) {
            delete vertex;
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

        /// @name Data access methods
        /// {

        /// Get the vertex property given its descriptor.
        VertexProperty& operator[](VertexDescriptor descriptor)
        {
          return unwrapVertex(getVertex(std::move(descriptor)));
        }

        /// Get the vertex property given its descriptor.
        const VertexProperty& operator[](VertexDescriptor descriptor) const
        {
          return unwrapVertex(getVertex(std::move(descriptor)));
        }

        /// Get the edge property given its descriptor.
        virtual EdgeProperty& operator[](EdgeDescriptor descriptor)
        {
          return unwrapEdge(getEdge(std::move(descriptor)));
        }

        /// Get the edge property given its descriptor.
        virtual const EdgeProperty& operator[](EdgeDescriptor descriptor) const
        {
          return unwrapEdge(getEdge(std::move(descriptor)));
        }

        /// }

        /// Get the number of vertices.
        size_t verticesCount() const
        {
          return vertices.size();
        }

        /// Get the number of edges.
        size_t edgesCount() const
        {
          if (directed) {
            return edges.size();
          }

          return edges.size() / 2;
        }

        /// Add a vertex to the graph.
        /// The property is cloned and its lifetime is tied to the graph.
        VertexDescriptor addVertex(VertexProperty property)
        {
          auto* ptr = new Vertex(std::move(property));
          bool vertexInsertion = graph.addNode(*ptr);
          assert(vertexInsertion);
          return VertexDescriptor(this, ptr);
        }

        /// Get the vertices of the graph.
        llvm::iterator_range<VertexIterator> getVertices() const
        {
          return llvm::iterator_range<VertexIterator>(
              VertexIterator::begin(*this, graph),
              VertexIterator::end(*this, graph));
        }

        /// Get the vertices of the graph that match a certain property.
        ///
        /// @param visibilityFn  function determining whether a vertex should be considered or not
        /// @return iterable range of vertex descriptors
        llvm::iterator_range<FilteredVertexIterator> getVertices(
            std::function<bool(const VertexProperty&)> visibilityFn) const
        {
          auto filter = [=](VertexDescriptor descriptor) -> bool {
            return visibilityFn((*this)[descriptor]);
          };

          auto allVertices = getVertices();

          FilteredVertexIterator begin(allVertices.begin(), allVertices.end(), filter);
          FilteredVertexIterator end(allVertices.end(), allVertices.end(), filter);

          return llvm::iterator_range<FilteredVertexIterator>(begin, end);
        }

        /// Add an edge to the graph.
        /// The property is cloned and stored into the graph data structures.
        EdgeDescriptor addEdge(
            VertexDescriptor from,
            VertexDescriptor to,
            EdgeProperty property = EdgeProperty())
        {
          Vertex& src = getVertex(from);
          Vertex& dest = getVertex(to);

          // Allocate the property on the heap, so that it can be shared between
          // both the edges, in case of undirected graph.
          auto* edgeProperty = new EdgeProperty(std::move(property));
          edgeProperties.push_back(edgeProperty);

          auto* ptr = new Edge(dest, edgeProperty);
          edges.push_back(ptr);
          bool edgeInsertion = graph.connect(src, dest, *ptr);
          assert(edgeInsertion);

          if (!directed) {
            auto* inversePtr = new Edge(src, edgeProperty);
            edges.push_back(inversePtr);
            bool inverseEdgeInsertion = graph.connect(dest, src, *inversePtr);
            assert(inverseEdgeInsertion);
          }

          return EdgeDescriptor(this, from, to, ptr);
        }

        /// Get all the edges of the graph.
        /// If the graph is undirected, then the edge between two nodes is
        /// returned only once and its source / destination order is casual,
        /// as it is indeed conceptually irrelevant.
        llvm::iterator_range<EdgeIterator> getEdges() const
        {
          return llvm::iterator_range<EdgeIterator>(
              EdgeIterator::begin(*this, directed, graph),
              EdgeIterator::end(*this, directed, graph));
        }

        /// Get all the edges of the graph that match a certain property.
        /// If the graph is undirected, then the edge between two nodes is
        /// returned only once and its source / destination order is casual,
        /// as it is indeed conceptually irrelevant.
        ///
        /// @param visibilityFn  function determining whether an edge should be considered or not
        /// @return iterable range of edge descriptors
        llvm::iterator_range<FilteredEdgeIterator> getEdges(
            std::function<bool(const EdgeProperty&)> visibilityFn) const
        {
          auto filter = [=](EdgeDescriptor descriptor) -> bool {
            return visibilityFn((*this)[descriptor]);
          };

          auto allEdges = getEdges();

          FilteredEdgeIterator begin(allEdges.begin(), allEdges.end(), filter);
          FilteredEdgeIterator end(allEdges.end(), allEdges.end(), filter);

          return llvm::iterator_range<FilteredEdgeIterator>(begin, end);
        }

        /// Get the edges exiting from a node.
        llvm::iterator_range<IncidentEdgeIterator> getOutgoingEdges(VertexDescriptor vertex) const
        {
          return llvm::iterator_range<IncidentEdgeIterator>(
              IncidentEdgeIterator::begin(*this, graph, getVertex(vertex), vertex),
              IncidentEdgeIterator::end(*this, graph, getVertex(vertex), vertex));
        }

        /// Get the edges exiting from a node that match a certain property.
        ///
        /// @param vertex        source vertex
        /// @param visibilityFn  function determining whether an edge should be considered or not
        /// @return iterable range of edge descriptors
        llvm::iterator_range<FilteredIncidentEdgeIterator> getOutgoingEdges(
            VertexDescriptor vertex,
            std::function<bool(const EdgeProperty&)> visibilityFn) const
        {
          auto filter = [=](EdgeDescriptor descriptor) -> bool {
            return visibilityFn((*this)[descriptor]);
          };

          auto allEdges = getOutgoingEdges(vertex);

          FilteredIncidentEdgeIterator begin(allEdges.begin(), allEdges.end(), filter);
          FilteredIncidentEdgeIterator end(allEdges.end(), allEdges.end(), filter);

          return llvm::iterator_range<FilteredIncidentEdgeIterator>(begin, end);
        }

        /// Get the vertices connected to a node.
        llvm::iterator_range<LinkedVerticesIterator> getLinkedVertices(VertexDescriptor vertex) const
        {
          return llvm::iterator_range<LinkedVerticesIterator>(
              LinkedVerticesIterator::begin(*this, graph, getVertex(vertex)),
              LinkedVerticesIterator::end(*this, graph, getVertex(vertex)));
        }

        /// Get the vertices connected to a node that match a certain property.
        ///
        /// @param vertex        source vertex
        /// @param visibilityFn  function determining whether a vertex should be considered or not
        /// @return iterable range of vertex descriptors
        llvm::iterator_range<FilteredLinkedVerticesIterator> getLinkedVertices(
            VertexDescriptor vertex,
            std::function<bool(const VertexProperty&)> visibilityFn) const
        {
          auto filter = [=](VertexDescriptor descriptor) -> bool {
            return visibilityFn((*this)[descriptor]);
          };

          auto allVertices = getLinkedVertices(vertex);

          FilteredLinkedVerticesIterator begin(allVertices.begin(), allVertices.end(), filter);
          FilteredLinkedVerticesIterator end(allVertices.end(), allVertices.end(), filter);

          return llvm::iterator_range<FilteredLinkedVerticesIterator>(begin, end);
        }

        /// Split the graph into multiple independent ones, if possible.
        // TODO create a view, instead of copying the vertices
        std::vector<Derived> getDisjointSubGraphs() const
        {
          std::vector<Derived> result;

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
        /// Get a vertex as it is stored in the base graph.
        Vertex& getVertex(VertexDescriptor descriptor)
        {
          return *descriptor.value;
        }

        /// Get a vertex as it is stored in the base graph.
        const Vertex& getVertex(VertexDescriptor descriptor) const
        {
          return *descriptor.value;
        }

        /// Get the vertex property of a vertex.
        VertexProperty& unwrapVertex(Vertex& vertex)
        {
          return *vertex;
        }

        /// Get the vertex property of a vertex.
        const VertexProperty& unwrapVertex(const Vertex& vertex) const
        {
          return *vertex;
        }

        /// Get an edge as it is stored in the base graph.
        Edge& getEdge(EdgeDescriptor descriptor)
        {
          return *descriptor.value;
        }

        /// Get an edge as it is stored in the base graph.
        const Edge& getEdge(EdgeDescriptor descriptor) const
        {
          return *descriptor.value;
        }

        /// Get the edge property of an edge.
        EdgeProperty& unwrapEdge(Edge& edge)
        {
          return **edge;
        }

        /// Get the edge property of an edge.
        const EdgeProperty& unwrapEdge(const Edge& edge) const
        {
          return **edge;
        }

        bool directed;
        std::vector<Vertex*> vertices;
        std::vector<Edge*> edges;
        std::vector<EdgeProperty*> edgeProperties;
        typename Traits::Base graph;
    };
  }

  template<typename Derived, typename VP, typename EP>
  class Graph
  {
    private:
      using Impl = impl::Graph<Derived, VP, EP>;

    public:
      using VertexProperty = typename Impl::VertexProperty;
      using EdgeProperty = typename Impl::EdgeProperty;

      using VertexDescriptor = typename Impl::VertexDescriptor;
      using EdgeDescriptor = typename Impl::EdgeDescriptor;

      using VertexIterator = typename Impl::VertexIterator;
      using FilteredVertexIterator = typename Impl::FilteredVertexIterator;

      using EdgeIterator = typename Impl::EdgeIterator;
      using FilteredEdgeIterator = typename Impl::FilteredEdgeIterator;

      using IncidentEdgeIterator = typename Impl::IncidentEdgeIterator;
      using FilteredIncidentEdgeIterator = typename Impl::FilteredIncidentEdgeIterator;

      using LinkedVerticesIterator = typename Impl::LinkedVerticesIterator;
      using FilteredLinkedVerticesIterator = typename Impl::FilteredLinkedVerticesIterator;

      Graph(bool directed) : impl(std::make_shared<Impl>(directed))
      {
      }

      /// @name Forwarding methods
      /// {

      VertexProperty& operator[](VertexDescriptor descriptor)
      {
        return (*impl)[descriptor];
      }

      const VertexProperty& operator[](VertexDescriptor descriptor) const
      {
        return (*impl)[descriptor];
      }

      virtual EdgeProperty& operator[](EdgeDescriptor descriptor)
      {
        return (*impl)[descriptor];
      }

      virtual const EdgeProperty& operator[](EdgeDescriptor descriptor) const
      {
        return (*impl)[descriptor];
      }

      size_t verticesCount() const
      {
        return impl->verticesCount();
      }

      size_t edgesCount() const
      {
        return impl->edgesCount();
      }

      VertexDescriptor addVertex(VertexProperty property)
      {
        return impl->addVertex(std::move(property));
      }

      llvm::iterator_range<VertexIterator> getVertices() const
      {
        return impl->getVertices();
      }

      llvm::iterator_range<FilteredVertexIterator> getVertices(
          std::function<bool(const VertexProperty&)> visibilityFn) const
      {
        return impl->getVertices(std::move(visibilityFn));
      }

      EdgeDescriptor addEdge(VertexDescriptor from, VertexDescriptor to, EdgeProperty property = EdgeProperty())
      {
        return impl->addEdge(std::move(from), std::move(to), std::move(property));
      }

      llvm::iterator_range<EdgeIterator> getEdges() const
      {
        return impl->getEdges();
      }

      llvm::iterator_range<FilteredEdgeIterator> getEdges(
          std::function<bool(const EdgeProperty&)> visibilityFn) const
      {
        return impl->getEdges(std::move(visibilityFn));
      }

      llvm::iterator_range<IncidentEdgeIterator> getOutgoingEdges(VertexDescriptor vertex) const
      {
        return impl->getOutgoingEdges(std::move(vertex));
      }

      llvm::iterator_range<FilteredIncidentEdgeIterator> getOutgoingEdges(
          VertexDescriptor vertex,
          std::function<bool(const EdgeProperty&)> visibilityFn) const
      {
        return impl->getOutgoingEdges(std::move(vertex), std::move(visibilityFn));
      }

      llvm::iterator_range<LinkedVerticesIterator> getLinkedVertices(VertexDescriptor vertex) const
      {
        return impl->getLinkedVertices(std::move(vertex));
      }

      llvm::iterator_range<FilteredLinkedVerticesIterator> getOutgoingEdges(
          VertexDescriptor vertex,
          std::function<bool(const VertexProperty&)> visibilityFn) const
      {
        return impl->getOutgoingEdges(std::move(vertex), std::move(visibilityFn));
      }

      /// Split the graph into multiple independent ones, if possible.
      // TODO create a view, instead of copying the vertices
      std::vector<Derived> getDisjointSubGraphs() const
      {
        return impl->getDisjointSubGraphs();
      }

      /// }

    private:
      std::shared_ptr<Impl> impl;
  };

  /// Default edge property.
  class EmptyEdgeProperty
  {
  };

  /// Undirected graph.
  template<typename VP, typename EP = internal::EmptyEdgeProperty>
  class UndirectedGraph : public Graph<UndirectedGraph<VP, EP>, VP, EP>
  {
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
      using FilteredIncidentEdgeIterator = typename Base::FilteredIncidentEdgeIterator;

      UndirectedGraph() : Base(false)
      {
      }

      UndirectedGraph(const UndirectedGraph& other) = default;

      UndirectedGraph(UndirectedGraph&& other) = default;

      ~UndirectedGraph() = default;

      UndirectedGraph& operator=(const UndirectedGraph& other) = default;
  };

  /// Directed graph.
  template<typename VP, typename EP = internal::EmptyEdgeProperty>
  class DirectedGraph : public Graph<DirectedGraph<VP, EP>, VP, EP>
  {
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
