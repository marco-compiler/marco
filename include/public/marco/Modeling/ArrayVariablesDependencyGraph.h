#ifndef MARCO_MODELING_ARRAYVARIABLESDEPENDENCYGRAPH_H
#define MARCO_MODELING_ARRAYVARIABLESDEPENDENCYGRAPH_H

#include "marco/Modeling/Dependency.h"
#include "marco/Modeling/SingleEntryWeaklyConnectedDigraph.h"
#include "marco/Modeling/SCC.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/SCCIterator.h"

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

    private:
      mlir::MLIRContext* context;
      Graph graph;
      WritesMap writesMap;

    public:
      explicit ArrayVariablesDependencyGraph(mlir::MLIRContext* context)
          : context(context)
      {
      }

      [[nodiscard]] mlir::MLIRContext* getContext() const
      {
        assert(context != nullptr);
        return context;
      }

      /// @name Forwarded methods
      /// {

      Equation& getEquation(EquationDescriptor descriptor)
      {
        return graph[descriptor];
      }

      const Equation& getEquation(EquationDescriptor descriptor) const
      {
        return graph[descriptor];
      }

      EquationProperty& operator[](EquationDescriptor descriptor)
      {
        return getEquation(descriptor).getProperty();
      }

      const EquationProperty& operator[](EquationDescriptor descriptor) const
      {
        return getEquation(descriptor).getProperty();
      }

      auto equationsBegin() const
      {
        return graph.verticesBegin();
      }

      auto equationsEnd() const
      {
        return graph.verticesEnd();
      }

      /// }

      void addEquations(llvm::ArrayRef<EquationProperty> equations)
      {
        // Add the equations to the graph and determine which equation writes
        // into which variable, together with the accessed indices.
        std::vector<EquationDescriptor> vertices =
            addEquationsAndMapWrites(equations);

        // Now that the writes are known, we can explore the reads in order to
        // determine the dependencies among the equations. An equation e1
        // depends on another equation e2 if e1 reads (a part) of a variable
        // that is written by e2. In this case, an arc from e2 to e1 is
        // inserted (meaning that e2 must be computed before e1).
        std::mutex graphMutex;

        auto mapFn = [&](EquationDescriptor equationDescriptor) {
          const Equation& equation = graph[equationDescriptor];
          std::vector<Access> reads = equation.getReads();

          for (const Access& read : reads) {
            IndexSet readIndices = read.getAccessFunction().map(
                equation.getIterationRanges());

            auto writeInfos = writesMap.equal_range(read.getVariable());

            for (const auto& [variableId, writeInfo] :
                 llvm::make_range(writeInfos.first, writeInfos.second)) {
              const IndexSet& writtenIndices =
                  writeInfo.getWrittenVariableIndexes();

              if (writtenIndices.overlaps(readIndices)) {
                std::lock_guard<std::mutex> lockGuard(graphMutex);
                graph.addEdge(writeInfo.getEquation(), equationDescriptor);
              }
            }
          }
        };

        mlir::parallelForEach(getContext(), vertices, mapFn);
      }

      /// Get all the SCCs.
      std::vector<SCC> getSCCs() const
      {
        std::vector<SCC> result;

        for (auto scc = llvm::scc_begin(&graph), end = llvm::scc_end(&graph);
             scc != end; ++scc) {
          std::vector<EquationDescriptor> equations;

          for (const auto& equation: *scc) {
            equations.push_back(equation);
          }

          // Ignore the entry node
          if (equations.size() > 1 || equations[0] != graph.getEntryNode()) {
            result.emplace_back(graph, equations.begin(), equations.end());
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
          llvm::ArrayRef<EquationProperty> equations)
      {
        std::vector<EquationDescriptor> vertices;

        // Differentiate the mutexes to enable more parallelism.
        std::mutex graphMutex;
        std::mutex writesMapMutex;

        auto mapFn = [&](const EquationProperty& equationProperty) {
          std::unique_lock<std::mutex> graphLockGuard(graphMutex);

          EquationDescriptor descriptor =
              graph.addVertex(Equation(equationProperty));

          vertices.push_back(descriptor);
          const Equation& equation = graph[descriptor];
          graphLockGuard.unlock();

          const auto& write = equation.getWrite();
          const auto& accessFunction = write.getAccessFunction();

          // Determine the indices of the variable that are written by the
          // equation.
          IndexSet writtenIndices(
              accessFunction.map(equation.getIterationRanges()));

          std::unique_lock<std::mutex> writesMapLockGuard(writesMapMutex);

          writesMap.emplace(
              write.getVariable(),
              WriteInfo(
                  graph,
                  write.getVariable(),
                  descriptor,
                  std::move(writtenIndices)));

          writesMapLockGuard.unlock();
        };

        mlir::parallelForEach(getContext(), equations, mapFn);
        return vertices;
      }
  };
}

#endif // MARCO_MODELING_ARRAYVARIABLESDEPENDENCYGRAPH_H
