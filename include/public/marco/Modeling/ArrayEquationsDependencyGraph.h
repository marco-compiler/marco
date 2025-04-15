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
/// Graph storing the dependencies between array equations.
/// An edge from equation A to equation B is created if the computations inside
/// A depends on B, meaning that B needs to be computed first. The order of
/// computations is therefore given by a post-order visit of the graph.
template <typename VariableProperty, typename EquationProperty,
          typename Graph =
              internal::dependency::SingleEntryWeaklyConnectedDigraph<
                  internal::dependency::ArrayEquation<EquationProperty>>>
class ArrayEquationsDependencyGraph {
public:
  using Base = Graph;

  using Variable = internal::dependency::ArrayVariable<VariableProperty>;
  using VariableId = typename Variable::Id;
  using Equation = typename Graph::VertexProperty;

  using EquationDescriptor = typename Graph::VertexDescriptor;
  using AccessProperty = typename Equation::Access::Property;

  using Access =
      ::marco::modeling::dependency::Access<VariableProperty, AccessProperty>;

  using WriteInfo =
      internal::dependency::WriteInfo<Graph, VariableId, EquationDescriptor>;

  using WritesMap = std::multimap<VariableId, WriteInfo>;

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

  /// Get the writes map of all the equations added to the graph.
  const WritesMap &getWritesMap() const { return writesMap; }

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
      std::vector<Access> writeAccesses = equation.getWrites();
      IndexSet writtenIndices;

      if (!writeAccesses.empty()) {
        for (const auto &writeAccess : writeAccesses) {
          const auto &accessFunction = writeAccess.getAccessFunction();
          writtenIndices += accessFunction.map(equation.getIterationRanges());
        }

        result.emplace(writeAccesses[0].getVariable(),
                       WriteInfo(*graph, writeAccesses[0].getVariable(), *it,
                                 std::move(writtenIndices)));
      }
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
    std::vector<Access> writeAccesses = equation.getWrites();
    IndexSet writtenIndices;

    if (!writeAccesses.empty()) {
      for (const Access &writeAccess : writeAccesses) {
        const auto &accessFunction = writeAccess.getAccessFunction();
        writtenIndices += accessFunction.map(equation.getIterationRanges());
      }

      std::unique_lock<std::mutex> writesMapLockGuard(writesMapMutex);

      writesMap.emplace(writeAccesses[0].getVariable(),
                        WriteInfo(*graph, writeAccesses[0].getVariable(),
                                  equationDescriptor,
                                  std::move(writtenIndices)));
    }
  }

  /// Explore the read accesses in order to determine the dependencies among the
  /// equations. An equation e1 depends on another equation e2 if e1 reads (a
  /// part) of a variable that is written by e2. In this case, an arc from e1
  /// to e2 is inserted (meaning that e1 needs the result of e2).
  template <typename EquationsIt>
  void addEdges(EquationsIt equationsBeginIt, EquationsIt equationsEndIt) {
    std::mutex graphMutex;

    auto mapFn = [&](EquationDescriptor equationDescriptor) {
      std::unique_lock<std::mutex> equationLockGuard(graphMutex);
      const Equation &equation = (*graph)[equationDescriptor];
      equationLockGuard.unlock();

      std::vector<Access> reads = equation.getReads();
      llvm::MapVector<VariableId, IndexSet> readVariables;

      // First collect the read indices, so that only one edge can be later
      // created in case of multiple accesses to the same variable.
      for (const Access &read : reads) {
        IndexSet readIndices =
            read.getAccessFunction().map(equation.getIterationRanges());

        readVariables[read.getVariable()] += readIndices;
      }

      // Create the edges.
      for (const auto &readEntry : readVariables) {
        const VariableId &variableId = readEntry.first;
        const IndexSet &readIndices = readEntry.second;

        auto writeInfos = writesMap.equal_range(variableId);

        for (const auto &[variableId, writeInfo] :
             llvm::make_range(writeInfos.first, writeInfos.second)) {
          const IndexSet &writtenIndices =
              writeInfo.getWrittenVariableIndexes();

          if (writtenIndices.overlaps(readIndices)) {
            std::lock_guard<std::mutex> edgeLockGuard(graphMutex);
            graph->addEdge(equationDescriptor, writeInfo.getEquation());
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
