#ifndef MARCO_MODELING_SCC_H
#define MARCO_MODELING_SCC_H

#include <list>
#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/SCCIterator.h>
#include <marco/utils/TreeOStream.h>
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

  namespace internal::scc
  {
    /**
     * Wrapper for variables.
     * Used to provide some utility methods.
     */
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

        Id getId() const
        {
          return property.getId();
        }

      private:
        // Custom variable property
        VariableProperty property;
    };

    /**
     * Wrapper for equations.
     * Used to provide some utility methods.
     */
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

    /**
     * Keeps track of which variable, together with its indexes, are written by an equation.
     */
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

    template<typename Graph, typename EquationDescriptor, typename Access>
    class DFSStep : public Dumpable
    {
      public:
        DFSStep(const Graph& graph, EquationDescriptor equation, MCIS equationIndexes, Access read)
          : graph(&graph),
            equation(std::move(equation)),
            equationIndexes(std::move(equationIndexes)),
            read(std::move(read))
        {
        }

        using Dumpable::dump;

        void dump(std::ostream& stream) const override
        {
          using namespace marco::utils;

          TreeOStream os(stream);
          os << "DFS step\n";
          os << tree_property << "Written variable: " << (*graph)[equation].getWrite().getVariable() << "\n";
          os << tree_property << "Writing equation: " << (*graph)[equation].getId() << "\n";
          os << tree_property << "Filtered equation indexes: " << equationIndexes << "\n";
          os << tree_property << "Read access: " << read.getAccessFunction() << "\n";
        }

        EquationDescriptor getEquation() const
        {
          return equation;
        }

        const MCIS& getEquationIndexes() const
        {
          return equationIndexes;
        }

        void setEquationIndexes(MCIS indexes)
        {
          equationIndexes = std::move(indexes);
        }

        const Access& getRead() const
        {
          return read;
        }

      private:
        const Graph* graph;
        EquationDescriptor equation;
        MCIS equationIndexes;
        Access read;
    };

    template<typename Graph, typename EquationDescriptor, typename Equation, typename Access>
    class FilteredEquation : public Dumpable
    {
      private:
        class Dependency : public Dumpable
        {
          public:
            Dependency(Access access, std::unique_ptr<FilteredEquation> equation)
              : access(std::move(access)), equation(std::move(equation))
            {
            }

            Dependency(const Dependency& other)
                : access(other.access), equation(std::make_unique<FilteredEquation>(*other.equation))
            {
            }

            Dependency(Dependency&& other) = default;

            ~Dependency() = default;

            friend void swap(Dependency& first, Dependency& second)
            {
              using std::swap;
              swap(first.access, second.access);
              swap(first.node, second.node);
            }

            Dependency& operator=(const Dependency& other)
            {
              Dependency result(other);
              swap(*this, result);
              return *this;
            }

            using Dumpable::dump;

            void dump(std::ostream& stream) const override
            {
              using namespace marco::utils;

              TreeOStream os(stream);
              os << "Access function: " << access.getAccessFunction() << "\n";
              os << tree_property;
              equation->dump(os);
            }

            const Access& getAccess() const
            {
              return access;
            }

            FilteredEquation& getNode()
            {
              assert(equation != nullptr);
              return *equation;
            }

            const FilteredEquation& getNode() const
            {
              assert(equation != nullptr);
              return *equation;
            }

          private:
            Access access;
            std::unique_ptr<FilteredEquation> equation;
        };

        class Interval : public Dumpable
        {
          public:
            using Container = std::vector<Dependency>;

          public:
            Interval(MultidimensionalRange range, llvm::ArrayRef<Dependency> destinations)
                : range(std::move(range)), destinations(destinations.begin(), destinations.end())
            {
            }

            Interval(MultidimensionalRange range, Access access, std::unique_ptr<FilteredEquation> destination)
                : range(std::move(range))
            {
              destinations.emplace_back(std::move(access), std::move(destination));
            }

            using Dumpable::dump;

            void dump(std::ostream& stream) const override
            {
              using namespace marco::utils;

              TreeOStream os(stream);
              os << tree_property << "Indexes: " << range << "\n";

              for (const auto& destination : destinations) {
                os << tree_property;
                destination.dump(os);
                os << "\n";
              }
            }

            const MultidimensionalRange& getRange() const
            {
              return range;
            }

            llvm::ArrayRef<Dependency> getDestinations() const
            {
              return destinations;
            }

            void addDestination(Access access, std::unique_ptr<FilteredEquation> destination)
            {
              destinations.emplace_back(std::move(access), std::move(destination));
            }

          private:
            MultidimensionalRange range;
            Container destinations;
        };

        using Container = std::vector<Interval>;

      public:
        using const_iterator = typename Container::const_iterator;

        FilteredEquation(const Graph& graph, EquationDescriptor equation)
          : graph(&graph), equation(std::move(equation))
        {
        }

        using Dumpable::dump;

        void dump(std::ostream& stream) const override
        {
          using namespace marco::utils;

          TreeOStream os(stream);
          os << "Equation (" <<  (*graph)[equation].getId() << ")\n";

          for (const auto& interval : intervals) {
            os << tree_property;
            interval.dump(os);
          }
        }

        /**
         * Get the equation property that is provided by the user.
         * The method has been created to serve as an API.
         *
         * @return equation property
         */
        const typename Equation::Property& getEquation() const
        {
          return (*graph)[equation].getProperty();
        }

        const_iterator begin() const
        {
          return intervals.begin();
        }

        const_iterator end() const
        {
          return intervals.end();
        }

        void addCyclicDependency(const std::list<DFSStep<Graph, EquationDescriptor, Access>>& steps)
        {
          addListIt(steps.begin(), steps.end());
        }

      private:
        template<typename It>
        void addListIt(It step, It end)
        {
          if (step == end)
            return;

          if (auto next = std::next(step); next != end) {
            Container newIntervals;
            MCIS range = step->getEquationIndexes();

            for (const auto& interval: intervals) {
              if (!range.overlaps(interval.getRange())) {
                newIntervals.push_back(interval);
                continue;
              }

              MCIS restrictedRanges(interval.getRange());
              restrictedRanges -= range;

              for (const auto& restrictedRange: restrictedRanges) {
                newIntervals.emplace_back(restrictedRange, interval.getDestinations());
              }

              for (const MultidimensionalRange& intersectingRange : range.intersect(interval.getRange())) {
                range -= intersectingRange;

                llvm::ArrayRef<Dependency> dependencies = interval.getDestinations();
                std::vector<Dependency> newDependencies(dependencies.begin(), dependencies.end());

                auto dependency = llvm::find_if(newDependencies, [&](const Dependency& dependency) {
                  return dependency.getNode().equation == step->getEquation();
                });

                if (dependency == newDependencies.end()) {
                  auto& newDependency = newDependencies.emplace_back(step->getRead(), std::make_unique<FilteredEquation>(*graph, next->getEquation()));
                  newDependency.getNode().addListIt(next, end);
                } else {
                  dependency->getNode().addListIt(next, end);
                }

                Interval newInterval(intersectingRange, newDependencies);
                newIntervals.push_back(std::move(newInterval));
              }
            }

            for (const auto& subRange: range) {
              std::vector<Dependency> dependencies;
              auto& dependency = dependencies.emplace_back(step->getRead(), std::make_unique<FilteredEquation>(*graph, next->getEquation()));
              dependency.getNode().addListIt(next, end);
              newIntervals.emplace_back(subRange, dependencies);
            }

            intervals = std::move(newIntervals);
          }
        }

        const Graph* graph;
        EquationDescriptor equation;
        Container intervals;
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

      using WriteInfo = internal::scc::WriteInfo<ConnectedGraph, typename Variable::Id, EquationDescriptor>;
      using WritesMap = std::multimap<typename Variable::Id, WriteInfo>;

      using DFSStep = internal::scc::DFSStep<ConnectedGraph, EquationDescriptor, Access>;
      using FilteredEquation = internal::scc::FilteredEquation<ConnectedGraph, EquationDescriptor, Equation, Access>;

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

      std::vector<FilteredEquation> getEquationsCycles() const
      {
        std::vector<FilteredEquation> result;

        for (const auto& graph: graphs) {
          for (auto scc: llvm::make_range(llvm::scc_begin(graph), llvm::scc_end(graph))) {
            auto writes = getWritesMap(graph, scc.begin(), scc.end());

            for (const auto& equationDescriptor : scc) {
              auto cycles = getEquationCyclicDependencies(graph, writes, equationDescriptor);
              FilteredEquation dependencies(graph, equationDescriptor);

              for (const auto& cycle : cycles) {
                dependencies.addCyclicDependency(cycle);
              }

              result.push_back(std::move(dependencies));
            }
          }
        }

        return result;
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

          // Determine the indexes of the variable that are written by the equation
          auto writtenIndexes = accessFunction.map(equation.getIterationRanges());

          result.emplace(write.getVariable(), WriteInfo(graph, write.getVariable(), *it, std::move(writtenIndexes)));
        }

        return result;
      }

      std::vector<std::list<DFSStep>> getEquationCyclicDependencies(
          const ConnectedGraph& graph,
          const WritesMap& writes,
          EquationDescriptor equation) const
      {
        std::vector<std::list<DFSStep>> cyclicPaths;
        std::stack<std::list<DFSStep>> stack;

        // The first equation starts with the full range, as it has no predecessors
        MCIS indexes(graph[equation].getIterationRanges());

        std::list<DFSStep> emptyPath;

        for (auto& extendedPath : appendReads(graph, emptyPath, equation, indexes)) {
          stack.push(std::move(extendedPath));
        }

        while (!stack.empty()) {
          auto& path = stack.top();

          std::vector<std::list<DFSStep>> extendedPaths;

          const auto& equationIndexes = path.back().getEquationIndexes();
          const auto& read = path.back().getRead();
          const auto& accessFunction = read.getAccessFunction();
          auto readIndexes = accessFunction.map(equationIndexes);

          // Get the equations writing into the read variable
          auto writeInfos = writes.equal_range(read.getVariable());

          for (const auto& [variableId, writeInfo] : llvm::make_range(writeInfos.first, writeInfos.second)) {
            const auto& writtenIndexes = writeInfo.getWrittenVariableIndexes();

            // If the ranges do not overlap, then there is no loop involving the writing equation
            if (!readIndexes.overlaps(writtenIndexes)) {
              continue;
            }

            auto intersection = readIndexes.intersect(writtenIndexes);
            EquationDescriptor writingEquation = writeInfo.getEquation();
            MCIS writingEquationIndexes(graph[writingEquation].getIterationRanges());

            auto usedWritingEquationIndexes = inverseAccessIndexes(
                writingEquationIndexes,
                graph[writingEquation].getWrite().getAccessFunction(),
                intersection);

            if (detectLoop(cyclicPaths, path, graph, writingEquation, usedWritingEquationIndexes)) {
              // Loop detected. It may either be a loop regarding the first variable or not. In any case, we should
              // stop visiting the tree, which would be infinite.
              continue;
            }

            for (auto& extendedPath : appendReads(graph, path, writingEquation, usedWritingEquationIndexes))
              extendedPaths.push_back(std::move(extendedPath));
          }

          stack.pop();

          for (auto& extendedPath : extendedPaths) {
            stack.push(std::move(extendedPath));
          }
        }

        return cyclicPaths;
      }

      std::vector<std::list<DFSStep>> appendReads(
          const ConnectedGraph& graph,
          const std::list<DFSStep>& path,
          EquationDescriptor equation,
          const MCIS& equationRange) const
      {
        std::vector<std::list<DFSStep>> result;

        for (const Access& read : graph[equation].getReads()) {
          std::list<DFSStep> extendedPath = path;
          extendedPath.emplace_back(graph, equation, equationRange, read);
          result.push_back(std::move(extendedPath));
        }

        return result;
      }

      /**
       * Detect whether adding a new equation with a given range would lead to a loop.
       * The path to be check is intentionally passed by copy, as its flow is restricted depending on the
       * equation to be added and such modification must not interfere with other paths.
       *
       * @param cyclicPaths      cyclic paths results list
       * @param path             candidate path
       * @param graph            graph to which the equations belong to
       * @param equation         equation that should be added to the path
       * @param equationIndexes  indexes of the equation to be added
       * @return true if the candidate equation would create a loop when added to the path; false otherwise
       */
      bool detectLoop(
          std::vector<std::list<DFSStep>>& cyclicPaths,
          std::list<DFSStep> path,
          const ConnectedGraph& graph,
          EquationDescriptor equation,
          const MCIS& equationIndexes) const
      {
        if (!path.empty()) {
          // Restrict the flow (starting from the end).

          auto previousWriteAccessFunction = graph[equation].getWrite().getAccessFunction();
          auto previouslyWrittenIndexes = previousWriteAccessFunction.map(equationIndexes);

          for (auto it = path.rbegin(); it != path.rend(); ++it) {
            const auto& readAccessFunction = it->getRead().getAccessFunction();
            it->setEquationIndexes(inverseAccessIndexes(it->getEquationIndexes(), readAccessFunction, previouslyWrittenIndexes));

            previousWriteAccessFunction = graph[it->getEquation()].getWrite().getAccessFunction();
            previouslyWrittenIndexes = previousWriteAccessFunction.map(it->getEquationIndexes());
          }

          // Search along the restricted path if the candidate equation has already been visited with the same indexes
          auto step = llvm::find_if(path, [&](const DFSStep& step) {
            return step.getEquation() == equation && step.getEquationIndexes() == equationIndexes;
          });

          if (step == path.begin()) {
            // We have found a loop involving the variable defined by the first equation. This is the kind of loops
            // we are interested to find, so add it to the results.

            cyclicPaths.push_back(std::move(path));
            return true;
          }

          // We have not found a loop for the variable of interest (that is, the one defined by the first equation),
          // but yet we can encounter loops among other equations. Thus, we need to identify them and stop traversing
          // the (infinite) tree.

          if (step != path.end()) {
            return true;
          }
        }

        return false;
      }

      /**
       * Apply the inverse of an access function to a set of indices.
       * If the access function is not invertible, then the inverse indexes are determined starting from a parent set.
       *
       * @param parentIndexes   parent index set
       * @param accessFunction  access function to be inverted and applied (if possible)
       * @param accessIndexes   indexes to be inverted
       * @return indexes mapping to accessIndexes when accessFunction is applied to them
       */
      MCIS inverseAccessIndexes(
          const MCIS& parentIndexes,
          const AccessFunction& accessFunction,
          const MCIS& accessIndexes) const
      {
        if (accessFunction.isInvertible()) {
          auto mapped = accessFunction.inverseMap(accessIndexes);
          assert(accessFunction.map(mapped).contains(accessIndexes));
          return mapped;
        }

        // If the access function is not invertible, then not all the iteration variables are
        // used. This loss of information don't allow to reconstruct the equation ranges that
        // leads to the dependency loop. Thus, we need to iterate on all the original equation
        // points and determine which of them lead to a loop. This is highly expensive but also
        // inevitable, and confined only to very few cases within real scenarios.

        MCIS result;

        for (const auto& range: parentIndexes) {
          for (const auto& point: range) {
            if (accessIndexes.contains(accessFunction.map(point))) {
              result += point;
            }
          }
        }

        return result;
      }

      std::vector<ConnectedGraph> graphs;
  };
}

#endif // MARCO_MODELING_SCC_H
