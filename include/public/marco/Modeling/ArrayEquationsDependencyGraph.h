#ifndef MARCO_MODELING_ARRAYEQUATIONSDEPENDENCYGRAPH_H
#define MARCO_MODELING_ARRAYEQUATIONSDEPENDENCYGRAPH_H

#include "marco/Modeling/Dependency.h"
#include "marco/Modeling/SCC.h"
#include "marco/Modeling/SingleEntryDigraph.h"
#include "marco/Modeling/SingleEntryWeaklyConnectedDigraph.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/SCCIterator.h"
#include <type_traits>

namespace marco::modeling {
template <typename VariableProperty, typename EquationProperty,
          typename Graph =
              internal::dependency::SingleEntryWeaklyConnectedDigraph<
                  internal::dependency::ArrayEquation<EquationProperty>>>
class ArrayEquationsDependencyGraph {
public:
  using Base = Graph;

  using Variable = internal::dependency::VariableWrapper<VariableProperty>;
  using Equation = typename Graph::VertexProperty;

  using EquationDescriptor = typename Graph::VertexDescriptor;
  using AccessProperty = typename Equation::Access::Property;

  using Access =
      ::marco::modeling::dependency::Access<VariableProperty, AccessProperty>;

  using WriteInfo =
      internal::dependency::WriteInfo<Graph, typename Variable::Id,
                                      EquationDescriptor>;
  using WritesMap = std::multimap<typename Variable::Id, WriteInfo>;

  using SCC = internal::dependency::SCC<Graph>;

private:
  mlir::MLIRContext *context;
  std::shared_ptr<Graph> graph;
  WritesMap writesMap;

public:
  template <
      typename G = Graph,
      typename = typename std::enable_if<std::is_same_v<
          G, internal::dependency::SingleEntryWeaklyConnectedDigraph<
                 typename G::VertexProperty, typename G::EdgeProperty>>>::type>
  explicit ArrayEquationsDependencyGraph(mlir::MLIRContext *context)
      : context(context) {
    graph = std::make_shared<Graph>();
  }

  ArrayEquationsDependencyGraph(mlir::MLIRContext *context,
                                std::shared_ptr<Graph> graph)
      : context(context), graph(graph) {
    mapWrites();
    addEdges(graph->verticesBegin(), graph->verticesEnd());
  }

  [[nodiscard]] mlir::MLIRContext *getContext() const {
    assert(context != nullptr);
    return context;
  }

  /// @name Forwarded methods
  /// {

  Equation &getEquation(EquationDescriptor descriptor) {
    return (*graph)[descriptor];
  }

  const Equation &getEquation(EquationDescriptor descriptor) const {
    return (*graph)[descriptor];
  }

  EquationProperty &operator[](EquationDescriptor descriptor) {
    return getEquation(descriptor).getProperty();
  }

  const EquationProperty &operator[](EquationDescriptor descriptor) const {
    return getEquation(descriptor).getProperty();
  }

  auto equationsBegin() const { return graph->verticesBegin(); }

  auto equationsEnd() const { return graph->verticesEnd(); }

  /// }

  void addEquations(llvm::ArrayRef<EquationProperty> equations) {
    // Add the equations to the graph and determine which equation writes
    // into which variable, together with the accessed indices.
    std::vector<EquationDescriptor> vertices =
        addEquationsAndMapWrites(equations);

    // Add the edges.
    addEdges(vertices.begin(), vertices.end());
  }

  /// Get all the SCCs.
  std::vector<SCC> getSCCs() const {
    std::vector<SCC> result;

    for (auto scc = llvm::scc_begin(&static_cast<const Graph &>(*graph)),
              end = llvm::scc_end(&static_cast<const Graph &>(*graph));
         scc != end; ++scc) {
      std::vector<EquationDescriptor> equations;

      for (const auto &equation : *scc) {
        equations.push_back(equation);
      }

      // Ignore the entry node
      if (equations.size() > 1 || equations[0] != graph->getEntryNode()) {
        result.emplace_back(*graph, equations.begin(), equations.end());
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
  template <typename It>
  WritesMap getWritesMap(It equationsBegin, It equationsEnd) const {
    WritesMap result;

    for (It it = equationsBegin; it != equationsEnd; ++it) {
      const auto &equation = (*graph)[*it];
      const auto &write = equation.getWrite();
      const auto &accessFunction = write.getAccessFunction();

      // Determine the indices of the variable that are written by the equation
      IndexSet writtenIndices(
          accessFunction.map(equation.getIterationRanges()));

      result.emplace(write.getVariable(),
                     WriteInfo(*graph, write.getVariable(), *it,
                               std::move(writtenIndices)));
    }

    return result;
  }

private:
  std::vector<EquationDescriptor>
  addEquationsAndMapWrites(llvm::ArrayRef<EquationProperty> equations) {
    std::vector<EquationDescriptor> vertices;

    // Differentiate the mutexes to enable more parallelism.
    std::mutex graphMutex;
    std::mutex writesMapMutex;

    auto mapFn = [&](const EquationProperty &equationProperty) {
      std::unique_lock<std::mutex> graphLockGuard(graphMutex);

      EquationDescriptor descriptor =
          graph->addVertex(Equation(equationProperty));

      vertices.push_back(descriptor);
      const Equation &equation = (*graph)[descriptor];
      graphLockGuard.unlock();

      mapWrite(writesMapMutex, equation, descriptor);
    };

    mlir::parallelForEach(getContext(), equations, mapFn);
    return vertices;
  }

  void mapWrites() {
    // Differentiate the mutexes to enable more parallelism.
    std::mutex graphMutex;
    std::mutex writesMapMutex;

    auto mapFn = [&](EquationDescriptor descriptor) {
      std::unique_lock<std::mutex> graphLockGuard(graphMutex);
      const Equation &equation = (*graph)[descriptor];
      graphLockGuard.unlock();

      mapWrite(writesMapMutex, equation, descriptor);
    };

    mlir::parallelForEach(getContext(), equationsBegin(), equationsEnd(),
                          mapFn);
  }

  void mapWrite(std::mutex &writesMapMutex, const Equation &equation,
                EquationDescriptor equationDescriptor) {
    const auto &write = equation.getWrite();
    const auto &accessFunction = write.getAccessFunction();

    // Determine the indices of the variable that are written by the
    // equation.
    IndexSet writtenIndices(accessFunction.map(equation.getIterationRanges()));

    std::unique_lock<std::mutex> writesMapLockGuard(writesMapMutex);

    writesMap.emplace(write.getVariable(),
                      WriteInfo(*graph, write.getVariable(), equationDescriptor,
                                std::move(writtenIndices)));
  }

  /// Explore the read accesses in order to determine the dependencies among the
  /// equations. An equation e1 depends on another equation e2 if e1 reads (a
  /// part) of a variable that is written by e2. In this case, an arc from e2
  /// to e1 is inserted (meaning that e2 must be computed before e1).
  template <typename EquationsIt>
  void addEdges(EquationsIt equationsBeginIt, EquationsIt equationsEndIt) {
    std::mutex graphMutex;

    auto mapFn = [&](EquationDescriptor equationDescriptor) {
      std::unique_lock<std::mutex> equationLockGuard(graphMutex);
      const Equation &equation = (*graph)[equationDescriptor];
      equationLockGuard.unlock();

      std::vector<Access> reads = equation.getReads();

      for (const Access &read : reads) {
        IndexSet readIndices =
            read.getAccessFunction().map(equation.getIterationRanges());

        auto writeInfos = writesMap.equal_range(read.getVariable());

        for (const auto &[variableId, writeInfo] :
             llvm::make_range(writeInfos.first, writeInfos.second)) {
          const IndexSet &writtenIndices =
              writeInfo.getWrittenVariableIndexes();

          if (writtenIndices.overlaps(readIndices)) {
            std::lock_guard<std::mutex> edgeLockGuard(graphMutex);
            graph->addEdge(writeInfo.getEquation(), equationDescriptor);
          }
        }
      }
    };

    mlir::parallelForEach(getContext(), equationsBeginIt, equationsEndIt,
                          mapFn);
  }
};
} // namespace marco::modeling

#endif // MARCO_MODELING_ARRAYEQUATIONSDEPENDENCYGRAPH_H
