#ifndef MARCO_MODELING_DEPENDENCY_H
#define MARCO_MODELING_DEPENDENCY_H

#include <list>
#include <llvm/ADT/GraphTraits.h>
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

        const typename VariableTraits<VariableProperty>::Id& getVariable() const
        {
          return variable;
        }

        const AccessFunction& getAccessFunction() const
        {
          return accessFunction;
        }

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
    /// Used to provide some utility methods.
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

    /// Wrapper for equations.
    /// Used to provide some utility methods.
    template<typename EquationProperty>
    class EquationVertex
    {
      public:
        using Property = EquationProperty;
        using Traits = ::marco::modeling::dependency::EquationTraits<EquationProperty>;
        using Id = typename Traits::Id;

        using Access = marco::modeling::dependency::Access<
            typename Traits::VariableType,
            typename internal::dependency::get_access_property<EquationProperty>::type>;

        EquationVertex(EquationProperty property)
            : property(property)
        {
        }

        bool operator==(const EquationVertex& other) const
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
          assert(index < getNumOfIterationVars());
          auto begin = Traits::getRangeBegin(&property, index);
          auto end = Traits::getRangeEnd(&property, index);
          return Range(begin, end);
        }

        MultidimensionalRange getIterationRanges() const
        {
          llvm::SmallVector<Range> ranges;

          for (size_t i = 0, e = getNumOfIterationVars(); i < e; ++i) {
            ranges.push_back(getIterationRange(i));
          }

          return MultidimensionalRange(ranges);
        }

        Access getWrite() const
        {
          return Traits::getWrite(&property);
        }

        std::vector<Access> getReads() const
        {
          return Traits::getReads(&property);
        }

      private:
        // Custom equation property
        EquationProperty property;
    };

    /// Keeps track of which variable, together with its indexes, are written by an equation.
    template<typename Graph, typename VariableId, typename EquationDescriptor>
    class WriteInfo : public Dumpable
    {
      public:
        WriteInfo(const Graph& graph, VariableId variable, EquationDescriptor equation, MultidimensionalRange indexes)
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

        const MultidimensionalRange& getWrittenVariableIndexes() const
        {
          return indexes;
        }

      private:
        // Used for debugging purpose
        const Graph* graph;
        VariableId variable;

        EquationDescriptor equation;
        MultidimensionalRange indexes;
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
    template<typename VertexProperty, typename EdgeProperty = EmptyEdgeProperty>
    class SingleEntryWeaklyConnectedDigraph
    {
      public:
        using Graph = DirectedGraph<PtrProperty<VertexProperty>, PtrProperty<EdgeProperty>>;

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

          assert(checkConsistency());
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
        bool checkConsistency() const
        {
          // TODO
          return true;
        }

        Graph graph;
        VertexDescriptor entryNode;
    };

    /// List of the equations composing an SCC.
    /// All the equations belong to a given graph.
    template<typename Graph>
    class SCC
    {
      public:
        //using Equation = typename Graph::VertexProperty;
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

namespace marco::modeling
{
  template<typename VariableProperty, typename EquationProperty>
  class VVarDependencyGraph
  {
    public:
      using Variable = internal::dependency::VariableWrapper<VariableProperty>;
      using Equation = internal::dependency::EquationVertex<EquationProperty>;

      // In order to search for SCCs we need to provide an entry point to the graph and the graph itself must
      // not be disjoint. We achieve this by creating a fake entry point that is connected to all the nodes.
      using Graph = internal::dependency::SingleEntryWeaklyConnectedDigraph<Equation>;

      using EquationDescriptor = typename Graph::VertexDescriptor;
      using AccessProperty = typename Equation::Access::Property;
      using Access = dependency::Access<VariableProperty, AccessProperty>;

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

      Equation& operator[](EquationDescriptor descriptor)
      {
        return graph[descriptor];
      }

      const Equation& operator[](EquationDescriptor descriptor) const
      {
        return graph[descriptor];
      }

      /// Get all the SCCs.
      std::vector<SCC> getSCCs() const
      {
        std::vector<SCC> result;

        for (auto scc = llvm::scc_begin(graph), end = llvm::scc_end(graph); scc != end; ++scc) {
          std::vector<EquationDescriptor> equations;

          for (const auto& equation : *scc) {
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
          auto writtenIndexes = accessFunction.map(equation.getIterationRanges());

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
      //using Graph = internal::dependency::SingleEntryWeaklyConnectedDigraph<Equation>;

      using Graph = internal::DirectedGraph<SCC>;
      using Equation = internal::dependency::EquationVertex<typename SCC::EquationProperty>;

      SCCDependencyGraph(llvm::ArrayRef<SCC> SCCs)
      {
        Graph graph;

        // Add the SCCs to the graph
        for (const auto& scc : SCCs) {
          graph.addVertex(scc);
        }

        for (const auto& scc : SCCs) {
          for (const auto& equationDescriptor : scc) {
            const auto& equation = scc.getGraph()[equationDescriptor];
            auto reads = equation.getReads();


          }
        }

        for (const auto& subGraph : graph.getDisjointSubGraphs()) {
          graphs.push_back(subGraph);
        }
      }

    private:
      Graph graphs;
  };

  template<typename VariableProperty, typename EquationProperty>
  class SVarDependencyGraph
  {
    // TODO
  };
}


#endif // MARCO_MODELING_DEPENDENCY_H
