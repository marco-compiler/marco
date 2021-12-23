#ifndef MARCO_MODELING_VVARDEPENDENCYGRAPH_H
#define MARCO_MODELING_VVARDEPENDENCYGRAPH_H

#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/SCCIterator.h>
#include <stack>

#include "AccessFunction.h"
#include "Dumpable.h"
#include "Graph.h"
#include "MCIS.h"
#include "MultidimensionalRange.h"

namespace marco::modeling
{
  namespace internal::scc
  {
    class EmptyAccessProperty
    {
    };

    template<class T>
    struct get_access_property
    {
      template<class U, typename = typename U::AccessProperty>
      static typename U::AccessProperty property(int);

      template<class U>
      static EmptyAccessProperty property(...);

      using type = decltype(property<T>(0));
    };
  }

  namespace scc
  {
    template<typename VariableProperty, typename AccessProperty = internal::scc::EmptyAccessProperty>
    class Access
    {
      public:
        using Property = AccessProperty;

        Access(VariableProperty variable, AccessFunction accessFunction, AccessProperty property = AccessProperty())
            : variable(std::move(variable)),
              accessFunction(std::move(accessFunction)),
              property(std::move(property))
        {
        }

        const VariableProperty& getVariable() const
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
        VariableProperty variable;
        AccessFunction accessFunction;
        AccessProperty property;
    };
  }

  namespace internal::scc
  {
    template<typename VariableProperty>
    class VariableWrapper
    {
      public:
        using Id = typename VariableProperty::Id;
        using Property = VariableProperty;

        VariableWrapper(VariableProperty property)
            : property(property)
        {
        }

        bool operator==(const VariableWrapper& other) const
        {
          return getId() == other.getId();
        }

        VariableWrapper::Id getId() const
        {
          return property.getId();
        }

        size_t getRank() const
        {
          return getRank(property);
        }

        long getDimensionSize(size_t index) const
        {
          return getDimensionSize(property, index);
        }

      private:
        // Custom equation property
        VariableProperty property;
    };

    template<class EquationProperty, class VariableProperty>
    class EquationVertex
    {
      public:
        using Id = typename EquationProperty::Id;
        using Property = EquationProperty;

        using Access = marco::modeling::scc::Access<
            VariableProperty,
            typename internal::scc::get_access_property<EquationProperty>::type>;

        EquationVertex(EquationProperty property)
            : property(property)
        {
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
          return property.getId();
        }

        size_t getNumOfIterationVars() const
        {
          return getNumOfIterationVars(property);
        }

        Range getIterationRange(size_t index) const
        {
          return getIterationRange(property, index);
        }

        MultidimensionalRange getIterationRanges() const
        {
          return getIterationRanges(property);
        }

        Access getWrite() const
        {
          return getWrite(property);
        }

        void getReads(llvm::SmallVectorImpl<Access>& accesses) const
        {
          getReads(property, accesses);
        }

      private:
        static size_t getNumOfIterationVars(const EquationProperty& p)
        {
          return p.getNumOfIterationVars();
        }

        static Range getIterationRange(const EquationProperty& p, size_t index)
        {
          assert(index < getNumOfIterationVars(p));
          return Range(p.getRangeStart(index), p.getRangeEnd(index));
        }

        static MultidimensionalRange getIterationRanges(const EquationProperty& p)
        {
          llvm::SmallVector<Range, 3> ranges;

          for (unsigned int i = 0, e = getNumOfIterationVars(p) ; i < e ; ++i) {
            ranges.push_back(getIterationRange(p, i));
          }

          return MultidimensionalRange(ranges);
        }

        static void getReads(
            const EquationProperty& p,
            llvm::SmallVectorImpl<Access>& accesses)
        {
          p.getReads(accesses);
        }

        static Access getWrite(const EquationProperty& p)
        {
          return p.getWrite();
        }

        // Custom equation property
        EquationProperty property;
    };

    template<typename EquationDescriptor>
    class WriteInfo
    {
      public:
        WriteInfo(EquationDescriptor equation, MultidimensionalRange writtenVariableIndexes)
            : equation(std::move(equation)), writtenVariableIndexes(std::move(writtenVariableIndexes))
        {
        }

        const EquationDescriptor& getEquation() const
        {
          return equation;
        }

        const MultidimensionalRange& getWrittenVariableIndexes() const
        {
          return writtenVariableIndexes;
        }

      private:
        EquationDescriptor equation;
        MultidimensionalRange writtenVariableIndexes;
    };

    template<typename Graph, typename EquationProperty, typename Access>
    class Dependence
    {
      public:
        Dependence(EquationProperty reader, EquationProperty writer,
                   MultidimensionalRange overlappingReadingEquationRange, Access access)
            : reader(std::move(reader)),
              writer(std::move(writer)),
              overlappingReadingEquationRange(std::move(overlappingReadingEquationRange)),
              access(std::move(access))
        {
        }

        const EquationProperty& getReader() const
        {
          return reader;
        }

        const EquationProperty& getWriter() const
        {
          return writer;
        }

        const MultidimensionalRange& getOverlappingReadingEquationRange()
        {
          return overlappingReadingEquationRange;
        }

        const Access& getAccess()
        {
          return access;
        }

      private:
        EquationProperty reader;
        EquationProperty writer;
        MultidimensionalRange overlappingReadingEquationRange;
        Access access;
    };

    template<typename Dependence>
    class SCC
    {
      public:
        SCC(llvm::ArrayRef<Dependence> dependencies) : dependencies(dependencies.begin(), dependencies.end())
        {
        }

        llvm::ArrayRef<Dependence> getDependencies() const
        {
          return dependencies;
        }

      private:
        llvm::SmallVector<Dependence, 3> dependencies;
    };

    class EmptyEdgeProperty
    {
    };

    template<typename VertexProperty, typename EdgeProperty>
    class DisjointDirectedGraph
    {
      public:
        using Graph = DirectedGraph<VertexProperty, EdgeProperty>;

        using VertexDescriptor = typename Graph::VertexDescriptor;
        using EdgeDescriptor = typename Graph::EdgeDescriptor;

        using VertexIterator = typename Graph::VertexIterator;
        using IncidentEdgeIterator = typename Graph::IncidentEdgeIterator;
        using LinkedVerticesIterator = typename Graph::LinkedVerticesIterator;

        DisjointDirectedGraph(Graph graph) : graph(std::move(graph))
        {
        }

        auto& operator[](VertexDescriptor vertex)
        {
          return graph[vertex];
        }

        const auto& operator[](VertexDescriptor vertex) const
        {
          return graph[vertex];
        }

        auto& operator[](EdgeDescriptor edge)
        {
          return graph[edge];
        }

        const auto& operator[](EdgeDescriptor edge) const
        {
          return graph[edge];
        }

        size_t size() const
        {
          return graph.verticesCount();
        }

        auto getVertices() const
        {
          return graph.getVertices();
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
    };
  }
}

namespace llvm
{
  // We specialize the LLVM's graph traits in order leverage the Tarjan algorithm
  // that is built into LLVM itself. This way we don't have to implement it from scratch.
  template<typename VertexProperty, typename EdgeProperty>
  struct GraphTraits<marco::modeling::internal::scc::DisjointDirectedGraph<VertexProperty, EdgeProperty>>
  {
    using Graph = marco::modeling::internal::scc::DisjointDirectedGraph<VertexProperty, EdgeProperty>;

    using NodeRef = typename Graph::VertexDescriptor;
    using ChildIteratorType = typename Graph::LinkedVerticesIterator;

    static NodeRef getEntryNode(const Graph& graph)
    {
      // Being the graph connected, we can safely treat any of its vertices
      // as entry node.
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
      using MultidimensionalRange = internal::MultidimensionalRange;
      using MCIS = internal::MCIS;

      using Variable = internal::scc::VariableWrapper<VariableProperty>;
      using Equation = internal::scc::EquationVertex<EquationProperty, VariableProperty>;

      using Edge = internal::scc::EmptyEdgeProperty;
      using Graph = internal::DirectedGraph<Equation, Edge>;
      using ConnectedGraph = internal::scc::DisjointDirectedGraph<Equation, Edge>;

      using EquationDescriptor = typename Graph::VertexDescriptor;
      using AccessProperty = typename Equation::Access::Property;
      using Access = scc::Access<VariableProperty, AccessProperty>;
      using WriteInfo = internal::scc::WriteInfo<EquationDescriptor>;

      using Dependence = internal::scc::Dependence<ConnectedGraph, Equation, Access>;
      using SCC = internal::scc::SCC<Dependence>;

      using WritesMap = std::multimap<typename VariableProperty::Id, WriteInfo>;

      VVarDependencyGraph(llvm::ArrayRef<EquationProperty> equations)
      {
        Graph graph;

        // Add the equations to the graph
        for (const auto& equationProperty : equations) {
          graph.addVertex(Equation(equationProperty));
        }

        // Determine which equation writes into which variable, together with the accessed indexes.
        auto vertices = graph.getVertices();
        auto writes = getWritesMap(graph, vertices.begin(), vertices.end());

        // Now that the writes are known, we can explore the reads in order to determine the dependencies among
        // the equations. An equation e1 depends on another equation e2 if e1 reads (a part) of a variable that is
        // written by e2.

        for (const auto& equationDescriptor : graph.getVertices()) {
          const Equation& equation = graph[equationDescriptor];

          llvm::SmallVector<Access> reads;
          equation.getReads(reads);

          for (const Access& read : reads) {
            auto readIndexes = read.getAccessFunction().map(equation.getIterationRanges());
            auto writeInfos = writes.equal_range(read.getVariable().getId());

            for (const auto&[variableId, writeInfo] : llvm::make_range(writeInfos.first, writeInfos.second)) {
              const auto& writtenIndexes = writeInfo.getWrittenVariableIndexes();

              if (writtenIndexes.overlaps(readIndexes)) {
                graph.addEdge(equationDescriptor, writeInfo.getEquation(), Edge());
              }
            }
          }
        }

        // In order to search for SCCs we need to provide an entry point to the graph.
        // However, the graph may be disjoint and thus only the SCCs reachable from the entry point would be
        // found. In order to avoid this, we split the graph into disjoint sub-graphs and later apply the Tarjan
        // algorithm on each of them.
        auto subGraphs = graph.getDisjointSubGraphs();

        for (const auto& subGraph : subGraphs) {
          graphs.emplace_back(std::move(subGraph));
        }
      }

      std::vector<SCC> getCircularDependencies()
      {
        std::vector<SCC> SCCs;

        for (const auto& graph : graphs) {
          for (auto scc : llvm::make_range(llvm::scc_begin(graph), llvm::scc_end(graph))) {
            llvm::SmallVector<Dependence, 3> dependencies;
            auto writes = getWritesMap(graph, scc.begin(), scc.end());

            for (const auto& equationDescriptor : scc) {
              const Equation& equation = graph[equationDescriptor];
              auto equationRange = graph[equationDescriptor].getIterationRanges();

              llvm::SmallVector<Access> reads;
              equation.getReads(reads);

              for (const Access& read : reads) {
                const auto& accessFunction = read.getAccessFunction();
                auto readIndexes = accessFunction.map(equationRange);
                auto writeInfos = writes.equal_range(read.getVariable().getId());

                for (const auto&[variableId, writeInfo] : llvm::make_range(writeInfos.first, writeInfos.second)) {
                  const auto& writtenIndexes = writeInfo.getWrittenVariableIndexes();

                  if (!readIndexes.overlaps(writtenIndexes)) {
                    continue;
                  }

                  auto intersection = readIndexes.intersect(writtenIndexes);

                  if (accessFunction.isInvertible()) {
                    dependencies.emplace_back(
                        graph[equationDescriptor],
                        graph[writeInfo.getEquation()],
                        accessFunction.inverseMap(intersection),
                        read);
                  } else {
                    MCIS mcis;

                    for (const auto& point : equationRange) {
                      if (intersection.contains(accessFunction.map(point))) {
                        mcis += point;
                      }
                    }

                    for (const auto& range : mcis) {
                      dependencies.emplace_back(graph[equationDescriptor], graph[writeInfo.getEquation()], range, read);
                    }
                  }
                }
              }
            }

            SCCs.emplace_back(std::move(dependencies));
          }
        }

        return SCCs;
      }

    private:
      template<typename Graph, typename It>
      WritesMap getWritesMap(const Graph& graph, It equationsBegin, It equationsEnd)
      {
        WritesMap result;

        for (It it = equationsBegin ; it != equationsEnd ; ++it) {
          const auto& equation = graph[*it];
          const auto& write = equation.getWrite();
          Variable writtenVariable(write.getVariable());
          const auto& accessFunction = write.getAccessFunction();
          auto writtenIndexes = accessFunction.map(equation.getIterationRanges());
          result.emplace(writtenVariable.getId(), WriteInfo(*it, std::move(writtenIndexes)));
        }

        return result;
      }

      std::vector<ConnectedGraph> graphs;
  };
}

#endif // MARCO_MODELING_VVARDEPENDENCYGRAPH_H
