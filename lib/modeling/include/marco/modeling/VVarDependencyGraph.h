#ifndef MARCO_MODELING_SCC_H
#define MARCO_MODELING_SCC_H

#include <list>
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
  namespace scc
  {
    // This class must be specialized for the variable type that is used during the loops identification process.
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

    // This class must be specialized for the equation type that is used during the matching process.
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

  namespace internal::scc
  {
    /**
     * Fallback access property, in case the user didn't provide one.
     */
    class EmptyAccessProperty
    {
    };

    template<class T>
    struct get_access_property
    {
      template<typename U>
      using Traits = ::marco::modeling::scc::EquationTraits<U>;

      template<class U, typename = typename Traits<U>::AccessProperty>
      static typename Traits<U>::AccessProperty property(int);

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

  namespace internal::scc
  {
    template<typename VariableProperty>
    class VariableWrapper
    {
      public:
        using Property = VariableProperty;
        using Traits = ::marco::modeling::scc::VariableTraits<VariableProperty>;
        using Id = typename Traits::Id;

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

      private:
        // Custom variable property
        VariableProperty property;
    };

    template<typename EquationProperty>
    class EquationVertex
    {
      public:
        using Property = EquationProperty;
        using Traits = ::marco::modeling::scc::EquationTraits<EquationProperty>;
        using Id = typename Traits::Id;

        using Access = marco::modeling::scc::Access<
            typename Traits::VariableType,
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
          llvm::SmallVector<Range, 3> ranges;

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

    template<typename EquationProperty, typename Access>
    class DependencyList
    {
      public:
        class Interval
        {
          private:
            using Destination = std::pair<Access, EquationProperty>;
            using Container = std::vector<Destination>;

          public:
            Interval(MultidimensionalRange range, llvm::ArrayRef<Destination> destinations)
                : range(std::move(range)), destinations(destinations.begin(), destinations.end())
            {
            }

            Interval(MultidimensionalRange range, Access access, EquationProperty destination)
                : range(std::move(range))
            {
              destinations.emplace_back(std::move(access), std::move(destination));
            }

            const MultidimensionalRange& getRange() const
            {
              return range;
            }

            llvm::ArrayRef<Destination> getDestinations() const
            {
              return destinations;
            }

            void addDestination(Access access, EquationProperty equation)
            {
              destinations.emplace_back(std::move(access), std::move(equation));
            }

          private:
            MultidimensionalRange range;
            Container destinations;
        };

        DependencyList(EquationProperty source)
            : source(std::move(source))
        {
        }

        const EquationProperty& getSource() const
        {
          return source;
        }

        void addDestination(MultidimensionalRange range, Access access, EquationProperty equation)
        {
          std::vector<Interval> newIntervals;
          MCIS mcis(range);

          for (const Interval& interval: intervals) {
            if (!interval.getRange().overlaps(range)) {
              continue;
            }

            mcis -= interval.getRange();

            for (const auto& subRange: interval.getRange().subtract(range)) {
              newIntervals.emplace_back(subRange, interval.getDestinations());
            }

            Interval newInterval(interval.getRange().intersect(range), interval.getDestinations());
            newInterval.addDestination(access, equation);
            newIntervals.push_back(std::move(newInterval));
          }

          for (const auto& subRange: mcis) {
            newIntervals.emplace_back(subRange, access, equation);
          }

          intervals = newIntervals;
        }

      private:
        EquationProperty source;
        std::vector<Interval> intervals;
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
        std::vector<Dependence> dependencies;
    };

    template<typename VertexProperty>
    class DisjointDirectedGraph
    {
      public:
        using Graph = DirectedGraph<VertexProperty>;

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

    template<typename VariableId, typename Equation, typename Access>
    class DFSStep
    {
      public:
        DFSStep(VariableId variable, Equation equation, MCIS range, Access access)
          : variable(std::move(variable)),
            equation(std::move(equation)),
            range(std::move(range)),
            access(std::move(access))
        {
        }

      //private:
        VariableId variable;
        Equation equation;
        MCIS range;
        Access access;
    };
  }
}

namespace llvm
{
  // We specialize the LLVM's graph traits in order leverage the Tarjan algorithm
  // that is built into LLVM itself. This way we don't have to implement it from scratch.
  template<typename VertexProperty>
  struct GraphTraits<marco::modeling::internal::scc::DisjointDirectedGraph<VertexProperty>>
  {
    using Graph = marco::modeling::internal::scc::DisjointDirectedGraph<VertexProperty>;

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
      using Equation = internal::scc::EquationVertex<EquationProperty>;

      using Graph = internal::DirectedGraph<Equation>;
      using ConnectedGraph = internal::scc::DisjointDirectedGraph<Equation>;

      using EquationDescriptor = typename Graph::VertexDescriptor;
      using AccessProperty = typename Equation::Access::Property;
      using Access = scc::Access<VariableProperty, AccessProperty>;
      using WriteInfo = internal::scc::WriteInfo<EquationDescriptor>;

      using DependencyList = internal::scc::DependencyList<Equation, Access>;
      using SCC = internal::scc::SCC<DependencyList>;

      using WritesMap = std::multimap<typename Variable::Id, WriteInfo>;
      using DFSStep = internal::scc::DFSStep<typename Variable::Id, Equation, Access>;

      VVarDependencyGraph(llvm::ArrayRef<EquationProperty> equations)
      {
        Graph graph;

        // Add the equations to the graph
        for (const auto& equationProperty: equations) {
          graph.addVertex(Equation(equationProperty));
        }

        // Determine which equation writes into which variable, together with the accessed indexes.
        auto vertices = graph.getVertices();
        auto writes = getWritesMap(graph, vertices.begin(), vertices.end());

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

        // In order to search for SCCs we need to provide an entry point to the graph.
        // However, the graph may be disjoint and thus only the SCCs reachable from the entry point would be
        // found. In order to avoid this, we split the graph into disjoint sub-graphs and later apply the Tarjan
        // algorithm on each of them.
        auto subGraphs = graph.getDisjointSubGraphs();

        for (const auto& subGraph: subGraphs) {
          graphs.emplace_back(std::move(subGraph));
        }
      }

      MCIS inverseAccessRange(
          const MCIS& parentRange,
          const AccessFunction& accessFunction,
          const MCIS& access) const
      {
        if (accessFunction.isInvertible()) {
          auto mapped = accessFunction.inverseMap(access);
          assert(accessFunction.map(mapped).contains(access));
          return mapped;
        }

        // If the access function is not invertible, then not all the iteration variables are
        // used. This loss of information don't allow to reconstruct the equation ranges that
        // leads to the dependency loop. Thus, we need to iterate on all the original equation
        // points and determine which of them lead to a loop. This is highly expensive but also
        // inevitable, and confined only to very few cases within real scenarios.

        MCIS result;

        for (const auto& range : parentRange) {
          for (const auto& point: range) {
            if (access.contains(accessFunction.map(point))) {
              result += point;
            }
          }
        }

        return result;
      }

      void processRead(
          std::vector<std::list<DFSStep>>& results,
          std::list<DFSStep> steps,
          const ConnectedGraph& graph,
          const WritesMap& writes,
          const Equation& equation,
          const MCIS& equationRange,
          const Access& read) const
      {
        steps.emplace_back(equation.getWrite().getVariable(), equation, equationRange, read);

        const auto& accessFunction = read.getAccessFunction();
        auto readIndexes = accessFunction.map(equationRange);

        // Get the equations writing into the read variable
        auto writeInfos = writes.equal_range(read.getVariable());

        for (const auto& [variableId, writeInfo] : llvm::make_range(writeInfos.first, writeInfos.second)) {
          const auto& writtenIndexes = writeInfo.getWrittenVariableIndexes();

          // If the ranges do not overlap, then there is no loop involving the writing equation
          if (!readIndexes.overlaps(writtenIndexes)) {
            continue;
          }

          auto intersection = readIndexes.intersect(writtenIndexes);
          const Equation& writingEquation = graph[writeInfo.getEquation()];
          MCIS writingEquationIndexes(writingEquation.getIterationRanges());

          auto usedWritingEquationIndexes = inverseAccessRange(
              writingEquationIndexes,
              writingEquation.getWrite().getAccessFunction(),
              intersection);

          processEquation(results, steps, graph, writes, writingEquation, usedWritingEquationIndexes);
        }
      }

      void processEquation(
          std::vector<std::list<DFSStep>>& results,
          std::list<DFSStep> steps,
          const ConnectedGraph& graph,
          const WritesMap& writes,
          const Equation& equation,
          const MCIS& equationRange) const
      {
        if (!steps.empty()) {
          if (steps.front().equation.getId() == equation.getId() && steps.front().range.contains(equationRange)) {
            // The first and current equation are the same and the first range contains the current one, so the path
            // is a loop candidate. Restrict the flow (starting from the end) and see if it holds true.

            auto previousWriteAccessFunction = equation.getWrite().getAccessFunction();
            auto previouslyWrittenIndexes = previousWriteAccessFunction.map(equationRange);

            for (auto it = steps.rbegin(); it != steps.rend(); ++it) {
              auto readAccessFunction = it->access.getAccessFunction();
              it->range = inverseAccessRange(it->range, readAccessFunction, previouslyWrittenIndexes);

              previousWriteAccessFunction = it->equation.getWrite().getAccessFunction();
              previouslyWrittenIndexes = previousWriteAccessFunction.map(it->range);
            }

            if (steps.front().range == equationRange) {
              // If the two ranges are the same, then a loop has been detected for what regards the variable defined
              // by the first equation.

              results.push_back(std::move(steps));
              return;
            }
          }

          // We have not found a loop for the variable of interest (that is, the one defined by the first equation),
          // but yet we can encounter loops among other equations. Thus, we need to identify them and stop traversing
          // the (infinite) tree. Two steps are considered to be equal if they traverse the same equation with the
          // same iteration indexes.

          auto equalStep = std::find_if(std::next(steps.rbegin()), steps.rend(), [&](const DFSStep& step) {
            return step.equation.getId() == equation.getId() && step.range == equationRange;
          });

          if (equalStep != steps.rend()) {
            return;
          }
        }

        // The reached equation does not lead to loops, so we can proceed visiting its children (that are the
        // equations it depends on).

        for (const Access& read : equation.getReads()) {
          processRead(results, steps, graph, writes, equation, equationRange, read);
        }
      }

      void processEquation(
          std::vector<std::list<DFSStep>>& results,
          const ConnectedGraph& graph,
          const WritesMap& writes,
          const Equation& equation) const
      {
        std::list<DFSStep> steps;

        // The first equation starts with the full range, as it has no predecessors
        MCIS range(equation.getIterationRanges());

        processEquation(results, steps, graph, writes, equation, range);
      }

      std::vector<SCC> getCircularDependencies() const
      {
        std::vector<SCC> SCCs;

        for (const auto& graph: graphs) {
          for (auto scc: llvm::make_range(llvm::scc_begin(graph), llvm::scc_end(graph))) {
            std::vector<DependencyList> dependencies;
            auto writes = getWritesMap(graph, scc.begin(), scc.end());

            for (const auto& equationDescriptor : scc) {
              const Equation& equation = graph[equationDescriptor];
              std::vector<std::list<DFSStep>> results;
              processEquation(results, graph, writes, equation);


              for (const auto& l : results) {
                std::cout << "SCC\n";

                for (const auto& step : l) {
                  std::cout << "id: " << step.equation.getId() << "\n";
                  std::cout << "range: " << step.range << "\n";
                  std::cout << "access: " << step.access.getAccessFunction() << "\n";
                }

                std::cout << "\n";
              }
            }
          }
        }

        return SCCs;
      }

    private:
      /**
       * Map each array variable to the equations that write into some of its scalar positions.
       *
       * @param graph           graph containing the equation
       * @param equationsBegin  beginning of the equations list
       * @param equationsEnd    ending of the equations list
       * @return variable - equations map
       */
      template<typename Graph, typename It>
      WritesMap getWritesMap(const Graph& graph, It equationsBegin, It equationsEnd) const
      {
        WritesMap result;

        for (It it = equationsBegin; it != equationsEnd; ++it) {
          const auto& equation = graph[*it];
          const auto& write = equation.getWrite();
          const auto& accessFunction = write.getAccessFunction();
          auto writtenIndexes = accessFunction.map(equation.getIterationRanges());
          result.emplace(write.getVariable(), WriteInfo(*it, std::move(writtenIndexes)));
        }

        return result;
      }

      std::vector<ConnectedGraph> graphs;
  };
}

#endif // MARCO_MODELING_SCC_H
