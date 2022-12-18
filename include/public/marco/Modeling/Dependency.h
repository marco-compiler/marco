#ifndef MARCO_MODELING_DEPENDENCY_H
#define MARCO_MODELING_DEPENDENCY_H

#include "marco/Diagnostic/TreeOStream.h"
#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/Dumpable.h"
#include "marco/Modeling/Graph.h"
#include "marco/Modeling/IndexSet.h"
#include "marco/Modeling/MultidimensionalRange.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ThreadPool.h"
#include <mutex>

namespace marco::modeling
{
  namespace dependency
  {
    // This class must be specialized for the variable type that is used during
    // the cycles identification process.
    template<typename VariableType>
    struct VariableTraits
    {
      // Elements to provide:
      //
      // typedef Id : the ID type of the variable.
      //
      // static Id getId(const VariableType*)
      //    return the ID of the variable.

      using Id = typename VariableType::UnknownVariableTypeError;
    };

    // This class must be specialized for the equation type that is used during
    // the cycles identification process.
    template<typename EquationType>
    struct EquationTraits
    {
      // Elements to provide:
      //
      // typedef Id : the ID type of the equation.
      //
      // static Id getId(const EquationType*)
      //    return the ID of the equation.
      //
      // static size_t getNumOfIterationVars(const EquationType*)
      //    return the number of induction variables.
      //
      // static MultidimensionalRange getIterationRanges(const EquationType*)
      //    return the iteration ranges.
      //
      // typedef VariableType : the type of the accessed variable
      //
      // typedef AccessProperty : the access property (this is optional, and if not specified an empty one is used)
      //
      // static Access<VariableType, AccessProperty> getWrite(const EquationType*)
      //    return the write access done by the equation.
      //
      // static std::vector<Access<VariableType, AccessProperty>> getReads(const EquationType*)
      //    return the read access done by the equation.

      using Id = typename EquationType::UnknownEquationTypeError;
    };

    template<typename SCC>
    struct SCCTraits
    {
      // Elements to provide:
      //
      // typedef ElementRef : the type of the elements composing the SCC, which should be cheap to copy.
      //
      // static bool hasCycle(const SCC* scc);
      //    return whether the SCC contains any cycle.
      //
      // static std::vector<ElementRef> getElements(const SCC* scc);
      //    return the elements composing the SCC.
      //
      // static std::vector<ElementRef> getDependencies(const Impl* SCC, ElementRef element);
      //    return the dependencies of an element, which may belong to other SCCs.
    };
  }

  namespace internal::dependency
  {
    /// Fallback access property, in case the user didn't provide one.
    class EmptyAccessProperty
    {
    };

    /// Determine the access property to be used according to the user-provided
    /// equation property.
    template<class T>
    struct get_access_property
    {
      template<typename U>
      using Traits = ::marco::modeling::dependency::EquationTraits<U>;

      template<class U, typename = typename Traits<U>::AccessProperty>
      static typename Traits<U>::AccessProperty property(int);

      template<class U>
      static EmptyAccessProperty property(...);

      using type = decltype(property<T>(0));
    };
  }

  namespace dependency
  {
    template<typename VariableProperty, typename AccessProperty = internal::dependency::EmptyAccessProperty>
    class Access
    {
      public:
        using Property = AccessProperty;

        Access(const VariableProperty& variable, AccessFunction accessFunction, AccessProperty property = AccessProperty())
            : variable(VariableTraits<VariableProperty>::getId(&variable)),
              accessFunction(std::move(accessFunction)),
              property(std::move(property))
        {
        }

        /// Get the ID of the accesses variable.
        const typename VariableTraits<VariableProperty>::Id& getVariable() const
        {
          return variable;
        }

        /// Get the access function.
        const AccessFunction& getAccessFunction() const
        {
          return accessFunction;
        }

        /// Get the user-defined access property.
        const AccessProperty& getProperty() const
        {
          return property;
        }

      private:
        typename VariableTraits<VariableProperty>::Id variable;
        AccessFunction accessFunction;
        AccessProperty property;
    };
  }

  namespace internal::dependency
  {
    /// Wrapper for variables.
    template<typename VariableProperty>
    class VariableWrapper
    {
      public:
        using Property = VariableProperty;
        using Traits = ::marco::modeling::dependency::VariableTraits<VariableProperty>;
        using Id = typename Traits::Id;

        VariableWrapper(VariableProperty property)
            : property(property)
        {
        }

        bool operator==(const VariableWrapper& other) const
        {
          return getId() == other.getId();
        }

        Id getId() const
        {
          return property.getId();
        }

      private:
        // Custom variable property
        VariableProperty property;
    };

    /// Utility class to provide additional methods relying on the ones provided by
    /// the user specialization.
    template<typename EquationProperty>
    class VectorEquationTraits
    {
      private:
        using Traits = ::marco::modeling::dependency::EquationTraits<EquationProperty>;

      public:
        using Id = typename Traits::Id;
        using VariableType = typename Traits::VariableType;
        using AccessProperty = typename get_access_property<EquationProperty>::type;

        /// @name Forwarding methods
        /// {

        static Id getId(const EquationProperty* equation)
        {
          return Traits::getId(equation);
        }

        static size_t getNumOfIterationVars(const EquationProperty* equation)
        {
          return Traits::getNumOfIterationVars(equation);
        }

        static IndexSet getIterationRanges(const EquationProperty* equation)
        {
          return Traits::getIterationRanges(equation);
        }

        using Access = ::marco::modeling::dependency::Access<VariableType, AccessProperty>;

        static Access getWrite(const EquationProperty* equation)
        {
          return Traits::getWrite(equation);
        }

        static std::vector<Access> getReads(const EquationProperty* equation)
        {
          return Traits::getReads(equation);
        }

        /// }
    };

    /// Wrapper for equations.
    template<typename EquationProperty>
    class VectorEquation
    {
      public:
        using Property = EquationProperty;
        using Traits = VectorEquationTraits<EquationProperty>;
        using Id = typename Traits::Id;
        using Access = typename Traits::Access;

        VectorEquation(EquationProperty property)
            : property(property)
        {
        }

        bool operator==(const VectorEquation& other) const
        {
          return getId() == other.getId();
        }

        EquationProperty& getProperty()
        {
          return property;
        }

        const EquationProperty& getProperty() const
        {
          return property;
        }

        /// @name Forwarding methods
        /// {

        Id getId() const
        {
          return Traits::getId(&property);
        }

        size_t getNumOfIterationVars() const
        {
          return Traits::getNumOfIterationVars(&property);
        }

        Range getIterationRange(size_t index) const
        {
          return Traits::getIterationRange(&property, index);
        }

        IndexSet getIterationRanges() const
        {
          return Traits::getIterationRanges(&property);
        }

        Access getWrite() const
        {
          return Traits::getWrite(&property);
        }

        std::vector<Access> getReads() const
        {
          return Traits::getReads(&property);
        }

        /// }

      private:
        // Custom equation property
        EquationProperty property;
    };

    /// Keeps track of which variable, together with its indexes, are written
    /// by an equation.
    template<typename Graph, typename VariableId, typename EquationDescriptor>
    class WriteInfo : public Dumpable
    {
      public:
        WriteInfo(const Graph& graph, VariableId variable, EquationDescriptor equation, IndexSet indexes)
            : graph(&graph), variable(std::move(variable)), equation(std::move(equation)), indexes(std::move(indexes))
        {
        }

        using Dumpable::dump;

        void dump(std::ostream& stream) const override
        {
          using namespace marco::utils;

          TreeOStream os(stream);
          os << "Write information\n";
          os << tree_property << "Variable: " << variable << "\n";
          os << tree_property << "Equation: " << (*graph)[equation].getId() << "\n";
          os << tree_property << "Written variable indexes: " << indexes << "\n";
        }

        const VariableId& getVariable() const
        {
          return variable;
        }

        EquationDescriptor getEquation() const
        {
          return equation;
        }

        const IndexSet& getWrittenVariableIndexes() const
        {
          return indexes;
        }

      private:
        // Used for debugging purpose
        const Graph* graph;
        VariableId variable;

        EquationDescriptor equation;
        IndexSet indexes;
    };

    template<typename Property>
    class PtrProperty
    {
      public:
        PtrProperty() : property(nullptr)
        {
        }

        explicit PtrProperty(Property property) : property(std::make_unique<Property>(std::move(property)))
        {
        }

        bool empty() const
        {
          return property == nullptr;
        }

        Property& operator*()
        {
          assert(!empty());
          return *property;
        }

        const Property& operator*() const
        {
          assert(!empty());
          return *property;
        }

      private:
        std::unique_ptr<Property> property;
    };

    /// A weakly connected directed graph with only one entry point.
    /// In other words, a directed graph that has an undirected path between
    /// any pair of vertices and has only one node with no predecessors.
    /// This is needed to work with the LLVM graph iterators, because a
    /// non-connected graph would lead to a visit of only the sub-graph
    /// containing the entry node.
    /// The single entry point also ensure the visit of all the nodes.
    /// The entry point is hidden from iteration upon vertices and can be
    /// accessed only by means of its dedicated getter.
    template<typename VP, typename EP = EmptyEdgeProperty>
    class SingleEntryWeaklyConnectedDigraph
    {
      public:
        using VertexProperty = VP;
        using EdgeProperty = EP;

      private:
        using Graph = DirectedGraph<PtrProperty<VertexProperty>, PtrProperty<EdgeProperty>>;

      public:
        using VertexDescriptor = typename Graph::VertexDescriptor;
        using EdgeDescriptor = typename Graph::EdgeDescriptor;

        using VertexIterator = typename Graph::VertexIterator;
        using IncidentEdgeIterator = typename Graph::IncidentEdgeIterator;
        using LinkedVerticesIterator = typename Graph::LinkedVerticesIterator;

        SingleEntryWeaklyConnectedDigraph()
            : entryNode(graph.addVertex(PtrProperty<VertexProperty>()))
        {
        }

        VertexProperty& operator[](VertexDescriptor vertex)
        {
          assert(vertex != entryNode && "The entry node doesn't have a property");
          return *graph[vertex];
        }

        const VertexProperty& operator[](VertexDescriptor vertex) const
        {
          assert(vertex != entryNode && "The entry node doesn't have a property");
          return *graph[vertex];
        }

        EdgeProperty& operator[](EdgeDescriptor edge)
        {
          return *graph[edge];
        }

        const EdgeProperty& operator[](EdgeDescriptor edge) const
        {
          return *graph[edge];
        }

        size_t size() const
        {
          return graph.verticesCount();
        }

        VertexDescriptor getEntryNode() const
        {
          return entryNode;
        }

        VertexDescriptor addVertex(VertexProperty property)
        {
          auto descriptor = graph.addVertex(PtrProperty(std::move(property)));

          // Connect the entry node to the new vertex
          graph.addEdge(entryNode, descriptor, PtrProperty<EdgeProperty>());

          return descriptor;
        }

        auto verticesBegin() const
        {
          return graph.verticesBegin([](const typename Graph::VertexProperty& vertex) {
            // Hide the entry point
            return !vertex.empty();
          });
        }

        auto verticesEnd() const
        {
          return graph.verticesEnd([](const typename Graph::VertexProperty& vertex) {
            // Hide the entry point
            return !vertex.empty();
          });
        }

        EdgeDescriptor addEdge(VertexDescriptor from, VertexDescriptor to, EdgeProperty property = EdgeProperty())
        {
          return graph.addEdge(from, to, PtrProperty(std::move(property)));
        }

        auto getEdges() const
        {
          return graph.getEdges();
        }

        auto outgoingEdgesBegin(VertexDescriptor vertex) const
        {
          return graph.outgoingEdgesBegin(std::move(vertex));
        }

        auto outgoingEdgesEnd(VertexDescriptor vertex) const
        {
          return graph.outgoingEdgesEnd(std::move(vertex));
        }

        auto linkedVerticesBegin(VertexDescriptor vertex) const
        {
          return graph.linkedVerticesBegin(std::move(vertex));
        }

        auto linkedVerticesEnd(VertexDescriptor vertex) const
        {
          return graph.linkedVerticesEnd(std::move(vertex));
        }

      private:
        Graph graph;
        VertexDescriptor entryNode;
    };

    /// List of the equations composing an SCC.
    /// All the equations belong to a given graph.
    template<typename Graph>
    class SCC
    {
      public:
        using EquationDescriptor = typename Graph::VertexDescriptor;

      private:
        using Equation = typename Graph::VertexProperty;
        using Container = std::vector<EquationDescriptor>;

      public:
        using iterator = typename Container::iterator;
        using const_iterator = typename Container::const_iterator;

        template<typename It>
        SCC(const Graph& graph, bool cycle, It equationsBegin, It equationsEnd)
          : graph(&graph), cycle(cycle), equations(equationsBegin, equationsEnd)
        {
        }

        const Graph& getGraph() const
        {
          assert(graph != nullptr);
          return *graph;
        }

        /// Get whether the SCC present a cycle.
        /// Note that only SCCs with just one element may not have cycles.
        bool hasCycle() const
        {
          return cycle;
        }

        /// Get the number of equations composing the SCC.
        size_t size() const
        {
          return equations.size();
        }

        const Equation& operator[](size_t index) const
        {
          assert(index < equations.size());
          return (*this)[equations[index]];
        }

        /// @name Forwarded methods
        /// {

        const Equation& operator[](EquationDescriptor descriptor) const
        {
          return (*graph)[descriptor];
        }

        /// }
        /// @name Iterators
        /// {

        iterator begin()
        {
          return equations.begin();
        }

        const_iterator begin() const
        {
          return equations.begin();
        }

        iterator end()
        {
          return equations.end();
        }

        const_iterator end() const
        {
          return equations.end();
        }

        /// }

      private:
        const Graph* graph;
        bool cycle;
        Container equations;
    };
  }

  namespace dependency
  {
    // Traits specialization for the internal SCC class
    template<typename Graph>
    class SCCTraits<internal::dependency::SCC<Graph>>
    {
      private:
        using Impl = internal::dependency::SCC<Graph>;

      public:
        using ElementRef = typename Impl::EquationDescriptor;

        static bool hasCycle(const Impl* SCC)
        {
          return SCC->hasCycle();
        }

        static std::vector<ElementRef> getElements(const Impl* SCC)
        {
          std::vector<ElementRef> result(SCC->begin(), SCC->end());
          return result;
        }

        static std::vector<ElementRef> getDependencies(const Impl* SCC, ElementRef element)
        {
          std::vector<ElementRef> result;
          const auto& graph = SCC->getGraph();

          auto edges = llvm::make_range(
              graph.outgoingEdgesBegin(element),
              graph.outgoingEdgesEnd(element));

          for (const auto& edge : edges) {
            result.push_back(edge.to);
          }

          return result;
        }
    };
  }
}

namespace llvm
{
  // We specialize the LLVM's graph traits in order leverage the algorithms
  // that are defined inside LLVM itself. This way we don't have to implement
  // them from scratch.
  template<typename VertexProperty>
  struct GraphTraits<const marco::modeling::internal::dependency::SingleEntryWeaklyConnectedDigraph<VertexProperty>*>
  {
    // The LLVM traits require the class specified as Graph to be copyable.
    // We use its address to overcome this limitation.
    using Graph = const marco::modeling::internal::dependency::SingleEntryWeaklyConnectedDigraph<VertexProperty>;
    using GraphPtr = Graph*;

    using NodeRef = typename Graph::VertexDescriptor;
    using ChildIteratorType = typename Graph::LinkedVerticesIterator;

    static NodeRef getEntryNode(const GraphPtr& graph)
    {
      return graph->getEntryNode();
    }

    static ChildIteratorType child_begin(NodeRef node)
    {
      return node.graph->linkedVerticesBegin(node);
    }

    static ChildIteratorType child_end(NodeRef node)
    {
      return node.graph->linkedVerticesEnd(node);
    }

    using nodes_iterator = typename Graph::VertexIterator;

    static nodes_iterator nodes_begin(GraphPtr* graph)
    {
      return (*graph)->verticesBegin();
    }

    static nodes_iterator nodes_end(GraphPtr* graph)
    {
      return (*graph)->verticesEnd();
    }

    using EdgeRef = typename Graph::EdgeDescriptor;
    using ChildEdgeIteratorType = typename Graph::IncidentEdgeIterator;

    static ChildEdgeIteratorType child_edge_begin(NodeRef node)
    {
      return node.graph->outgoingEdgesBegin(node);
    }

    static ChildEdgeIteratorType child_edge_end(NodeRef node)
    {
      return node.graph->outgoingEdgesEnd(node);
    }

    static NodeRef edge_dest(EdgeRef edge)
    {
      return edge.to;
    }

    static size_t size(GraphPtr* graph)
    {
      return (*graph)->size();
    }
  };
}

namespace marco::modeling
{
  template<typename VariableProperty, typename EquationProperty>
  class ArrayVariablesDependencyGraph
  {
    public:
      using Variable = internal::dependency::VariableWrapper<VariableProperty>;
      using Equation = internal::dependency::VectorEquation<EquationProperty>;

      // In order to search for SCCs we need to provide an entry point to the
      // graph and the graph itself must not be disjoint. We achieve this by
      // creating a fake entry point that is connected to all the nodes.
      using Graph = internal::dependency::SingleEntryWeaklyConnectedDigraph<Equation>;

      using EquationDescriptor = typename Graph::VertexDescriptor;
      using AccessProperty = typename Equation::Access::Property;
      using Access = ::marco::modeling::dependency::Access<VariableProperty, AccessProperty>;

      using WriteInfo = internal::dependency::WriteInfo<Graph, typename Variable::Id, EquationDescriptor>;
      using WritesMap = std::multimap<typename Variable::Id, WriteInfo>;

      using SCC = internal::dependency::SCC<Graph>;

      /// @name Forwarded methods
      /// {

      Equation& operator[](EquationDescriptor descriptor)
      {
        return graph[descriptor];
      }

      const Equation& operator[](EquationDescriptor descriptor) const
      {
        return graph[descriptor];
      }

      /// }

      void addEquations(llvm::ArrayRef<EquationProperty> equations)
      {
        llvm::ThreadPool threadPool;

        // Add the equations to the graph and determine which equation writes
        // into which variable, together with the accessed indices.
        std::vector<EquationDescriptor> vertices = addEquationsAndMapWrites(threadPool, equations);

        // Now that the writes are known, we can explore the reads in order to
        // determine the dependencies among the equations. An equation e1
        // depends on another equation e2 if e1 reads (a part) of a variable
        // that is written by e2.
        size_t numOfNewVertices = vertices.size();
        std::atomic_size_t currentVertex = 0;

        std::mutex graphMutex;

        auto mapFn = [&]() {
          size_t i = currentVertex++;

          while (i < numOfNewVertices) {
            const EquationDescriptor& equationDescriptor = vertices[i];
            const Equation& equation = graph[equationDescriptor];
            std::vector<Access> reads = equation.getReads();

            for (const Access& read : reads) {
              IndexSet readIndices = read.getAccessFunction().map(equation.getIterationRanges());
              auto writeInfos = writesMap.equal_range(read.getVariable());

              for (const auto& [variableId, writeInfo]: llvm::make_range(writeInfos.first, writeInfos.second)) {
                const IndexSet& writtenIndices = writeInfo.getWrittenVariableIndexes();

                if (writtenIndices.overlaps(readIndices)) {
                  std::lock_guard<std::mutex> lockGuard(graphMutex);
                  graph.addEdge(equationDescriptor, writeInfo.getEquation());
                }
              }
            }

            i = currentVertex++;
          }
        };

        // Shard the work among multiple threads.
        unsigned int numOfThreads = threadPool.getThreadCount();
        llvm::ThreadPoolTaskGroup tasks(threadPool);

        for (unsigned int i = 0; i < numOfThreads; ++i) {
          tasks.async(mapFn);
        }

        // Wait for all the threads to terminate.
        tasks.wait();
      }

      /// Get all the SCCs.
      std::vector<SCC> getSCCs() const
      {
        std::vector<SCC> result;

        for (auto scc = llvm::scc_begin(&graph), end = llvm::scc_end(&graph); scc != end; ++scc) {
          std::vector<EquationDescriptor> equations;

          for (const auto& equation: *scc) {
            equations.push_back(equation);
          }

          // Ignore the entry node
          if (equations.size() > 1 || equations[0] != graph.getEntryNode()) {
            result.emplace_back(graph, scc.hasCycle(), equations.begin(), equations.end());
          }
        }

        return result;
      }

      /// Map each array variable to the equations that write into some of its
      /// scalar positions.
      ///
      /// @param equationsBegin  beginning of the equations list
      /// @param equationsEnd    ending of the equations list
      /// @return variable - equations map
      template<typename It>
      WritesMap getWritesMap(It equationsBegin, It equationsEnd) const
      {
        WritesMap result;

        for (It it = equationsBegin; it != equationsEnd; ++it) {
          const auto& equation = graph[*it];
          const auto& write = equation.getWrite();
          const auto& accessFunction = write.getAccessFunction();

          // Determine the indices of the variable that are written by the equation
          IndexSet writtenIndices(accessFunction.map(equation.getIterationRanges()));

          result.emplace(write.getVariable(), WriteInfo(graph, write.getVariable(), *it, std::move(writtenIndices)));
        }

        return result;
      }

    private:
      std::vector<EquationDescriptor> addEquationsAndMapWrites(
        llvm::ThreadPool& threadPool,
        llvm::ArrayRef<EquationProperty> equations)
      {
        std::vector<EquationDescriptor> vertices;

        // Differentiate the mutexes to enable more parallelism.
        std::mutex graphMutex;
        std::mutex writesMapMutex;

        size_t numOfEquations = equations.size();
        std::atomic_size_t currentEquation = 0;

        auto mapFn = [&]() {
          size_t i = currentEquation++;

          while (i < numOfEquations) {
            const EquationProperty& equationProperty = equations[i];

            std::unique_lock<std::mutex> graphLockGuard(graphMutex);
            EquationDescriptor descriptor = graph.addVertex(Equation(equationProperty));
            vertices.push_back(descriptor);
            const Equation& equation = graph[descriptor];
            graphLockGuard.unlock();

            const auto& write = equation.getWrite();
            const auto& accessFunction = write.getAccessFunction();

            // Determine the indices of the variable that are written by the equation
            IndexSet writtenIndices(accessFunction.map(equation.getIterationRanges()));

            std::unique_lock<std::mutex> writesMapLockGuard(writesMapMutex);
            writesMap.emplace(write.getVariable(), WriteInfo(graph, write.getVariable(), descriptor, std::move(writtenIndices)));
            writesMapLockGuard.unlock();

            i = currentEquation++;
          }
        };

        // Shard the work among multiple threads.
        unsigned int numOfThreads = threadPool.getThreadCount();
        llvm::ThreadPoolTaskGroup tasks(threadPool);

        for (unsigned int i = 0; i < numOfThreads; ++i) {
          tasks.async(mapFn);
        }

        // Wait for all the threads to terminate.
        tasks.wait();

        return vertices;
      }

    private:
      Graph graph;
      WritesMap writesMap;
  };

  template<typename SCC>
  class SCCDependencyGraph
  {
    public:
      using Graph = internal::dependency::SingleEntryWeaklyConnectedDigraph<SCC>;
      using SCCDescriptor = typename Graph::VertexDescriptor;
      using SCCTraits = typename ::marco::modeling::dependency::SCCTraits<SCC>;
      using ElementRef = typename SCCTraits::ElementRef;

      /// @name Forwarded methods
      /// {

      SCC& operator[](SCCDescriptor descriptor)
      {
        return graph[descriptor];
      }

      const SCC& operator[](SCCDescriptor descriptor) const
      {
        return graph[descriptor];
      }

      /// }

      void addSCCs(llvm::ArrayRef<SCC> SCCs)
      {
        // Internalize the SCCs and keep track of the parent-children
        // relationships.
        llvm::DenseMap<ElementRef, SCCDescriptor> parentSCC;

        for (const auto& scc : SCCs) {
          auto sccDescriptor = graph.addVertex(scc);

          for (const auto& element : SCCTraits::getElements(&graph[sccDescriptor])) {
            parentSCC.try_emplace(element, sccDescriptor);
          }
        }

        // Connect the SCCs
        for (const auto& sccDescriptor : llvm::make_range(graph.verticesBegin(), graph.verticesEnd())) {
          const auto& scc = graph[sccDescriptor];

          // The set of SCCs that have already been connected to the current
          // SCC. This allows to avoid duplicate edges.
          llvm::DenseSet<SCCDescriptor> connectedSCCs;

          for (const auto& equationDescriptor : scc) {
            for (const auto& destination : SCCTraits::getDependencies(&scc, equationDescriptor)) {
              auto destinationSCC = parentSCC.find(destination)->second;

              if (!connectedSCCs.contains(destinationSCC)) {
                graph.addEdge(sccDescriptor, destinationSCC);
                connectedSCCs.insert(destinationSCC);
              }
            }
          }
        }
      }

      /// Perform a post-order visit of the dependency graph
      /// and get the ordered SCC descriptors.
      std::vector<SCCDescriptor> postOrder() const
      {
        std::vector<SCCDescriptor> result;
        std::set<SCCDescriptor> set;

        for (SCCDescriptor scc : llvm::post_order_ext(&graph, set)) {
          // Ignore the entry node
          if (scc != graph.getEntryNode()) {
            result.push_back(scc);
          }
        }

        return result;
      }

    private:
      Graph graph;
  };

  namespace internal::dependency
  {
    /// An equation defined on a single (multidimensional) index.
    /// Differently from the vector equation, this does not have dedicated
    /// traits. This is because the class itself is made for internal usage
    /// and all the needed information by applying the vector equation traits
    /// on the equation property. In other words, this class is used just to
    /// restrict the indices upon a vector equation iterates.
    template<typename EquationProperty>
    class ScalarEquation
    {
      public:
        using Property = EquationProperty;
        using VectorEquationTraits = ::marco::modeling::dependency::EquationTraits<EquationProperty>;
        using Id = typename VectorEquationTraits::Id;

        ScalarEquation(EquationProperty property, Point index)
          : property(std::move(property)), index(std::move(index))
        {
        }

        Id getId() const
        {
          return VectorEquationTraits::getId(&property);
        }

        const EquationProperty& getProperty() const
        {
          return property;
        }

        const Point& getIndex() const
        {
          return index;
        }

      private:
        EquationProperty property;
        Point index;
    };
  }

  template<typename VariableProperty, typename EquationProperty>
  class ScalarVariablesDependencyGraph
  {
    public:
      using VectorEquationTraits = ::marco::modeling::internal::dependency::VectorEquationTraits<EquationProperty>;

      using Variable = internal::dependency::VariableWrapper<VariableProperty>;
      using ScalarEquation = internal::dependency::ScalarEquation<EquationProperty>;

      using Graph = internal::dependency::SingleEntryWeaklyConnectedDigraph<ScalarEquation>;

      using ScalarEquationDescriptor = typename Graph::VertexDescriptor;
      using AccessProperty = typename VectorEquationTraits::AccessProperty;
      using Access = ::marco::modeling::dependency::Access<VariableProperty, AccessProperty>;

      using WriteInfo = internal::dependency::WriteInfo<Graph, typename Variable::Id, ScalarEquationDescriptor>;
      using WritesMap = std::multimap<typename Variable::Id, WriteInfo>;

      /// @name Forwarded methods
      /// {

      ScalarEquation& operator[](ScalarEquationDescriptor descriptor)
      {
        return graph[descriptor];
      }

      const ScalarEquation& operator[](ScalarEquationDescriptor descriptor) const
      {
        return graph[descriptor];
      }

      /// }

      void addEquations(llvm::ArrayRef<EquationProperty> equations)
      {
        llvm::ThreadPool threadPool;
        unsigned int numOfThreads = threadPool.getThreadCount();

        // Add the equations to the graph, while keeping track of which scalar
        // equation writes into each scalar variable.
        std::vector<ScalarEquationDescriptor> vertices = addEquationsAndMapWrites(threadPool, equations);

        // Determine the dependencies among the equations.
        size_t numOfNewVertices = vertices.size();
        std::atomic_size_t currentVertex = 0;

        std::mutex graphMutex;

        auto mapFn = [&]() {
          size_t i = currentVertex++;

          while (i < numOfNewVertices) {
            const ScalarEquationDescriptor& descriptor = vertices[i];
            const ScalarEquation& scalarEquation = graph[descriptor];
            auto reads = VectorEquationTraits::getReads(&scalarEquation.getProperty());

            for (const Access& read : reads) {
              auto readIndexes = read.getAccessFunction().map(scalarEquation.getIndex());
              auto writeInfos = writesMap.equal_range(read.getVariable());

              for (const auto& [variableId, writeInfo]: llvm::make_range(writeInfos.first, writeInfos.second)) {
                const auto& writtenIndexes = writeInfo.getWrittenVariableIndexes();

                if (writtenIndexes == readIndexes) {
                  std::lock_guard<std::mutex> lockGuard(graphMutex);
                  graph.addEdge(descriptor, writeInfo.getEquation());
                }
              }
            }

            i = currentVertex++;
          }
        };

        // Shard the work among multiple threads.
        llvm::ThreadPoolTaskGroup tasks(threadPool);

        for (unsigned int i = 0; i < numOfThreads; ++i) {
          tasks.async(mapFn);
        }

        // Wait for all the threads to terminate.
        tasks.wait();
      }

      /// Perform a post-order visit of the dependency graph and get the
      /// ordered scalar equation descriptors.
      std::vector<ScalarEquationDescriptor> postOrder() const
      {
        std::vector<ScalarEquationDescriptor> result;
        std::set<ScalarEquationDescriptor> set;

        for (ScalarEquationDescriptor equation : llvm::post_order_ext(&graph, set)) {
          // Ignore the entry node
          if (equation != graph.getEntryNode()) {
            result.push_back(equation);
          }
        }

        return result;
      }

    private:
      std::vector<ScalarEquationDescriptor> addEquationsAndMapWrites(
        llvm::ThreadPool& threadPool,
        llvm::ArrayRef<EquationProperty> equations)
      {
        std::vector<ScalarEquationDescriptor> vertices;

        // Differentiate the mutexes to enable more parallelism.
        std::mutex graphMutex;
        std::mutex writesMapMutex;

        size_t numOfEquations = equations.size();
        std::atomic_size_t currentEquation = 0;

        auto mapFn = [&]() {
          size_t i = currentEquation++;

          while (i < numOfEquations) {
            const EquationProperty& equationProperty = equations[i];
            const auto& write = VectorEquationTraits::getWrite(&equationProperty);
            const auto& accessFunction = write.getAccessFunction();

            for (const auto& equationIndices : VectorEquationTraits::getIterationRanges(&equationProperty)) {
              std::unique_lock<std::mutex> graphLockGuard(graphMutex);
              ScalarEquationDescriptor descriptor = graph.addVertex(ScalarEquation(equationProperty, equationIndices));
              vertices.push_back(descriptor);
              graphLockGuard.unlock();

              IndexSet writtenIndices(accessFunction.map(equationIndices));
              std::lock_guard<std::mutex> writesMapLockGuard(writesMapMutex);

              writesMap.emplace(
                  write.getVariable(),
                  WriteInfo(graph, write.getVariable(), descriptor, std::move(writtenIndices)));
            }

            i = currentEquation++;
          }
        };

        // Shard the work among multiple threads.
        unsigned int numOfThreads = threadPool.getThreadCount();
        llvm::ThreadPoolTaskGroup tasks(threadPool);

        for (unsigned int i = 0; i < numOfThreads; ++i) {
          tasks.async(mapFn);
        }

        // Wait for all the threads to terminate.
        tasks.wait();

        return vertices;
      }

    private:
      Graph graph;
      WritesMap writesMap;
  };
}

#endif // MARCO_MODELING_DEPENDENCY_H
