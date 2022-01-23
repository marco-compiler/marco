#ifndef MARCO_MODELING_DEPENDENCY_H
#define MARCO_MODELING_DEPENDENCY_H

#include <list>
#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SCCIterator.h>
#include <llvm/ADT/STLExtras.h>
#include <marco/utils/TreeOStream.h>
#include <stack>

#include "AccessFunction.h"
#include "Dumpable.h"
#include "Graph.h"
#include "MCIS.h"
#include "MultidimensionalRange.h"

namespace marco::modeling
{
  namespace dependency
  {
    // This class must be specialized for the variable type that is used during the cycles identification process.
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

    // This class must be specialized for the equation type that is used during the cycles identification process.
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
      // static long getRangeBegin(const EquationType*, size_t inductionVarIndex)
      //    return the beginning (included) of the range of an iteration variable.
      //
      // static long getRangeEnd(const EquationType*, size_t inductionVarIndex)
      //    return the ending (not included) of the range of an iteration variable.
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
        using Base = ::marco::modeling::dependency::EquationTraits<EquationProperty>;

      public:
        using Id = typename Base::Id;
        using VariableType = typename Base::VariableType;
        using AccessProperty = typename get_access_property<EquationProperty>::type;

        /// @name Forwarding methods
        /// {

        static Id getId(const EquationProperty* equation)
        {
          return Base::getId(equation);
        }

        static size_t getNumOfIterationVars(const EquationProperty* equation)
        {
          return Base::getNumOfIterationVars(equation);
        }

        static long getRangeBegin(const EquationProperty* equation, size_t inductionVarIndex)
        {
          return Base::getRangeBegin(equation, inductionVarIndex);
        }

        static long getRangeEnd(const EquationProperty* equation, size_t inductionVarIndex)
        {
          return Base::getRangeEnd(equation, inductionVarIndex);
        }

        using Access = ::marco::modeling::dependency::Access<VariableType, AccessProperty>;

        static Access getWrite(const EquationProperty* equation)
        {
          return Base::getWrite(equation);
        }

        static std::vector<Access> getReads(const EquationProperty* equation)
        {
          return Base::getReads(equation);
        }

        /// }
        /// @name Utility methods
        /// {

        /// Get the iteration range of a specific iteration variable.
        static Range getIterationRange(const EquationProperty* equation, size_t index)
        {
          assert(index < getNumOfIterationVars(equation));
          auto begin = getRangeBegin(equation, index);
          auto end = getRangeEnd(equation, index);
          return Range(begin, end);
        }

        /// Get the multidimensional iteration range of the whole equation.
        static MultidimensionalRange getIterationRanges(const EquationProperty* equation)
        {
          llvm::SmallVector<Range> ranges;

          for (size_t i = 0, e = getNumOfIterationVars(equation); i < e; ++i) {
            ranges.push_back(getIterationRange(equation, i));
          }

          return MultidimensionalRange(ranges);
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

        MultidimensionalRange getIterationRanges() const
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

    /// Keeps track of which variable, together with its indexes, are written by an equation.
    template<typename Graph, typename VariableId, typename EquationDescriptor>
    class WriteInfo : public Dumpable
    {
      public:
        WriteInfo(const Graph& graph, VariableId variable, EquationDescriptor equation, MCIS indexes)
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

        const MCIS& getWrittenVariableIndexes() const
        {
          return indexes;
        }

      private:
        // Used for debugging purpose
        const Graph* graph;
        VariableId variable;

        EquationDescriptor equation;
        MCIS indexes;
    };

    template<typename Property>
    class PtrProperty
    {
      public:
        PtrProperty() : property(nullptr)
        {
        }

        explicit PtrProperty(Property property) : property(std::make_shared<Property>(std::move(property)))
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
        // TODO convert to unique_ptr?
        std::shared_ptr<Property> property;
    };

    /// A weakly connected directed graph with only one entry point.
    /// In other words, a directed graph that has an undirected path between any pair
    /// of vertices and has only one node with no predecessors.
    /// This is needed to work with the LLVM graph iterators, because a non-connected
    /// graph would lead to a visit of only the sub-graph containing the entry node.
    /// The single entry point also ensure the visit of all the nodes.
    /// The entry point is hidden from iteration upon vertices and can be accessed
    /// only by means of its dedicated getter.
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

        SingleEntryWeaklyConnectedDigraph() : entryNode(graph.addVertex(PtrProperty<VertexProperty>()))
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

        auto getVertices() const
        {
          return graph.getVertices([](const typename Graph::VertexProperty& vertex) {
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

        auto getOutgoingEdges(VertexDescriptor vertex) const
        {
          return graph.getOutgoingEdges(std::move(vertex));
        }

        auto getLinkedVertices(VertexDescriptor vertex) const
        {
          return graph.getLinkedVertices(std::move(vertex));
        }

      private:
        Graph graph;
        VertexDescriptor entryNode;
    };

    /// List of the equations composing an SCC.
    /// All the equations belong to a given graph.
    template<typename G>
    class SCC
    {
      public:
        using Graph = G;
        using Equation = typename Graph::VertexProperty;
        using EquationDescriptor = typename Graph::VertexDescriptor;

      private:
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

        bool hasCycle() const
        {
          return cycle;
        }

        EquationDescriptor operator[](size_t index) const
        {
          assert(index < size());
          return equations[index];
        }

        size_t size() const
        {
          return equations.size();
        }

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

      private:
        const Graph* graph;
        bool cycle;
        Container equations;
    };
  }
}

namespace llvm
{
  // We specialize the LLVM's graph traits in order leverage the algorithms that are
  // defined inside LLVM itself. This way we don't have to implement them from scratch.
  template<typename VertexProperty>
  struct GraphTraits<marco::modeling::internal::dependency::SingleEntryWeaklyConnectedDigraph<VertexProperty>>
  {
    using Graph = marco::modeling::internal::dependency::SingleEntryWeaklyConnectedDigraph<VertexProperty>;

    using NodeRef = typename Graph::VertexDescriptor;
    using ChildIteratorType = typename Graph::LinkedVerticesIterator;

    static NodeRef getEntryNode(const Graph& graph)
    {
      return graph.getEntryNode();
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
      auto edges = node.graph->getOutgoingEdges(node);
      return edges.begin();
    }

    static ChildEdgeIteratorType child_edge_end(NodeRef node)
    {
      auto edges = node.graph->getOutgoingEdges(node);
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

namespace marco::modeling::internal
{
  template<typename VariableProperty, typename EquationProperty>
  class VVarDependencyGraph
  {
    public:
      using Variable = internal::dependency::VariableWrapper<VariableProperty>;
      using Equation = internal::dependency::VectorEquation<EquationProperty>;

      // In order to search for SCCs we need to provide an entry point to the graph and the graph itself must
      // not be disjoint. We achieve this by creating a fake entry point that is connected to all the nodes.
      using Graph = internal::dependency::SingleEntryWeaklyConnectedDigraph<Equation>;

      using EquationDescriptor = typename Graph::VertexDescriptor;
      using AccessProperty = typename Equation::Access::Property;
      using Access = ::marco::modeling::dependency::Access<VariableProperty, AccessProperty>;

      using WriteInfo = internal::dependency::WriteInfo<Graph, typename Variable::Id, EquationDescriptor>;
      using WritesMap = std::multimap<typename Variable::Id, WriteInfo>;

      using SCC = internal::dependency::SCC<Graph>;

      VVarDependencyGraph(llvm::ArrayRef<EquationProperty> equations)
      {
        // Add the equations to the graph
        for (const auto& equationProperty: equations) {
          graph.addVertex(Equation(equationProperty));
        }

        // Determine which equation writes into which variable, together with the accessed indexes.
        auto vertices = graph.getVertices();
        auto writes = getWritesMap(vertices.begin(), vertices.end());

        // Now that the writes are known, we can explore the reads in order to determine the dependencies among
        // the equations. An equation e1 depends on another equation e2 if e1 reads (a part) of a variable that is
        // written by e2.

        for (const auto& equationDescriptor: graph.getVertices()) {
          const Equation& equation = graph[equationDescriptor];

          auto reads = equation.getReads();

          for (const Access& read: reads) {
            auto readIndexes = read.getAccessFunction().map(equation.getIterationRanges());
            auto writeInfos = writes.equal_range(read.getVariable());

            for (const auto&[variableId, writeInfo]: llvm::make_range(writeInfos.first, writeInfos.second)) {
              const auto& writtenIndexes = writeInfo.getWrittenVariableIndexes();

              if (writtenIndexes.overlaps(readIndexes)) {
                graph.addEdge(equationDescriptor, writeInfo.getEquation());
              }
            }
          }
        }
      }

      /// @name Forwarding methods
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

      /// Get all the SCCs.
      std::vector<SCC> getSCCs() const
      {
        std::vector<SCC> result;

        for (auto scc = llvm::scc_begin(graph), end = llvm::scc_end(graph); scc != end; ++scc) {
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

      /// Map each array variable to the equations that write into some of its scalar positions.
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

          // Determine the indexes of the variable that are written by the equation
          MCIS writtenIndexes(accessFunction.map(equation.getIterationRanges()));

          result.emplace(write.getVariable(), WriteInfo(graph, write.getVariable(), *it, std::move(writtenIndexes)));
        }

        return result;
      }

      Graph graph;
  };

  template<typename SCC>
  class SCCDependencyGraph
  {
    public:
      using Graph = internal::dependency::SingleEntryWeaklyConnectedDigraph<SCC>;

      using Equation = typename SCC::Equation;
      using EquationDescriptor = typename SCC::EquationDescriptor;
      using SCCDescriptor = typename Graph::VertexDescriptor;

      SCCDependencyGraph(llvm::ArrayRef<SCC> SCCs)
      {
        // Keep track of the SCC an equation belongs to
        llvm::DenseMap<EquationDescriptor, SCCDescriptor> parentSCC;

        // Add the SCCs to the graph
        for (const auto& scc : SCCs) {
          auto sccDescriptor = graph.addVertex(scc);

          for (const auto& equationDescriptor : scc) {
            parentSCC.try_emplace(equationDescriptor, sccDescriptor);
          }
        }

        // Connect the SCCs
        for (const auto& sccDescriptor : graph.getVertices()) {
          const auto& scc = graph[sccDescriptor];
          const auto& originalGraph = scc.getGraph();

          // The set of SCCs that have already been connected to the current SCC.
          // This allows to avoid duplicate edges.
          llvm::DenseSet<SCCDescriptor> connectedSCCs;

          for (const auto& equationDescriptor : scc) {
            for (const auto& edgeDescriptor : originalGraph.getOutgoingEdges(equationDescriptor)) {
              auto destinationSCC = parentSCC.find(edgeDescriptor.to)->second;

              if (!connectedSCCs.contains(destinationSCC)) {
                graph.addEdge(sccDescriptor, destinationSCC);
                connectedSCCs.insert(destinationSCC);
              }
            }
          }
        }
      }

      /// @name Forwarding methods
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

      std::vector<SCCDescriptor> postOrder() const
      {
        std::vector<SCCDescriptor> result;
        std::set<SCCDescriptor> set;

        for (SCCDescriptor scc : llvm::post_order_ext(graph, set)) {
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
}

namespace marco::modeling::internal
{
  namespace dependency
  {
    /// An equation defined on a single (multidimensional) index.
    /// Differently from the vector equation, this does not have dedicated traits. This is because the class
    /// itself is made for internal usage and all the needed information by applying the vector equation traits
    /// on the equation property. In other words, this class is used just to restrict the indexes upon a vector
    /// equaion iterates.
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
  class SVarDependencyGraph
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

      SVarDependencyGraph(llvm::ArrayRef<EquationProperty> equations)
      {
        // Add the equations to the graph, while keeping track of which scalar equation
        // writes into each scalar variable.
        WritesMap writes;

        for (const auto& equationProperty: equations) {
          const auto& write = VectorEquationTraits::getWrite(&equationProperty);
          const auto& accessFunction = write.getAccessFunction();

          for (const auto& equationIndex : VectorEquationTraits::getIterationRanges(&equationProperty)) {
            auto scalarEquationDescriptor = graph.addVertex(ScalarEquation(equationProperty, equationIndex));
            MCIS writtenIndexes(accessFunction.map(equationIndex));

            writes.emplace(
                write.getVariable(),
                WriteInfo(graph, write.getVariable(), scalarEquationDescriptor, std::move(writtenIndexes)));
          }
        }

        // Determine the dependencies among the equations
        for (const auto& equationDescriptor: graph.getVertices()) {
          const ScalarEquation& scalarEquation = graph[equationDescriptor];
          auto reads = VectorEquationTraits::getReads(&scalarEquation.getProperty());

          for (const Access& read: reads) {
            auto readIndexes = read.getAccessFunction().map(scalarEquation.getIndex());
            auto writeInfos = writes.equal_range(read.getVariable());

            for (const auto& [variableId, writeInfo]: llvm::make_range(writeInfos.first, writeInfos.second)) {
              const auto& writtenIndexes = writeInfo.getWrittenVariableIndexes();

              if (writtenIndexes == readIndexes) {
                graph.addEdge(equationDescriptor, writeInfo.getEquation());
              }
            }
          }
        }
      }

      /// @name Forwarding methods
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

      std::vector<ScalarEquationDescriptor> postOrder() const
      {
        std::vector<ScalarEquationDescriptor> result;
        std::set<ScalarEquationDescriptor> set;

        for (ScalarEquationDescriptor equation : llvm::post_order_ext(graph, set)) {
          // Ignore the entry node
          if (equation != graph.getEntryNode()) {
            result.push_back(equation);
          }
        }

        return result;
      }

    private:
      Graph graph;
  };
}


#endif // MARCO_MODELING_DEPENDENCY_H
