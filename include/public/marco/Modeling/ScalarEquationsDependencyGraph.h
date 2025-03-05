#ifndef MARCO_MODELING_SCALAREQUATIONSDEPENDENCYGRAPH_H
#define MARCO_MODELING_SCALAREQUATIONSDEPENDENCYGRAPH_H

#include "marco/Modeling/Dependency.h"
#include "marco/Modeling/SingleEntryWeaklyConnectedDigraph.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/PostOrderIterator.h"
#include <mutex>
#include <set>
#include <type_traits>
#include <vector>

namespace marco::modeling {
namespace internal::dependency {
/// An equation defined on a single (multidimensional) index.
/// Differently from the vector equation, this does not have dedicated
/// traits. This is because the class itself is made for internal usage
/// and all the needed information by applying the vector equation traits
/// on the equation property. In other words, this class is used just to
/// restrict the indices upon a vector equation iterates.
template <typename EquationProperty>
class ScalarEquation {
public:
  using Property = EquationProperty;

  using VectorEquationTraits =
      ::marco::modeling::dependency::EquationTraits<EquationProperty>;

  using Id = typename VectorEquationTraits::Id;

  ScalarEquation(EquationProperty property, Point index)
      : property(std::move(property)), index(std::move(index)) {}

  Id getId() const { return VectorEquationTraits::getId(&property); }

  const EquationProperty &getProperty() const { return property; }

  const Point &getIndex() const { return index; }

private:
  EquationProperty property;
  Point index;
};
} // namespace internal::dependency

/// Graph storing the dependencies between scalar equations.
/// An edge from equation A to equation B is created if the computations inside
/// A depends on B, meaning that B needs to be computed first. The order of
/// computations is therefore given by a post-order visit of the graph.
template <typename VariableProperty, typename EquationProperty,
          typename Graph =
              internal::dependency ::SingleEntryWeaklyConnectedDigraph<
                  internal::dependency::ScalarEquation<EquationProperty>>>
class ScalarEquationsDependencyGraph {
public:
  using Base = Graph;

  using ArrayEquationTraits =
      typename ::marco::modeling::internal::dependency::ArrayEquation<
          EquationProperty>::Traits;

  using Variable = internal::dependency::ArrayVariable<VariableProperty>;
  using ScalarEquation = typename Graph::VertexProperty;

  using ScalarEquationDescriptor = typename Graph::VertexDescriptor;
  using AccessProperty = typename ArrayEquationTraits::AccessProperty;

  using Access =
      ::marco::modeling::dependency::Access<VariableProperty, AccessProperty>;

  using WriteInfo =
      internal::dependency::WriteInfo<Graph, typename Variable::Id,
                                      ScalarEquationDescriptor>;

  using WritesMap = std::multimap<typename Variable::Id, WriteInfo>;

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
  explicit ScalarEquationsDependencyGraph(mlir::MLIRContext *context)
      : context(context) {
    graph = std::make_shared<Graph>();
  }

  ScalarEquationsDependencyGraph(mlir::MLIRContext *context,
                                 std::shared_ptr<Graph> graph)
      : context(context), graph(graph) {
    mapWrites();
    addEdges(graph->verticesBegin(), graph->verticesEnd());
  }

  mlir::MLIRContext *getContext() const {
    assert(context != nullptr);
    return context;
  }

  /// @name Forwarded methods
  /// {

  ScalarEquation &operator[](ScalarEquationDescriptor descriptor) {
    return (*graph)[descriptor];
  }

  const ScalarEquation &operator[](ScalarEquationDescriptor descriptor) const {
    return (*graph)[descriptor];
  }

  /// }

  void addEquations(llvm::ArrayRef<EquationProperty> equations) {
    // Add the equations to the graph, while keeping track of which scalar
    // equation writes into each scalar variable.
    std::vector<ScalarEquationDescriptor> vertices =
        addEquationsAndMapWrites(equations);

    // Determine the dependencies among the equations.
    addEdges(vertices.begin(), vertices.end());
  }

  /// Perform a post-order visit of the dependency graph and get the
  /// ordered scalar equation descriptors.
  std::vector<ScalarEquationDescriptor> postOrder() const {
    std::vector<ScalarEquationDescriptor> result;
    std::set<ScalarEquationDescriptor> set;

    for (ScalarEquationDescriptor equation :
         llvm::post_order_ext(&static_cast<const Graph &>(*graph), set)) {
      // Ignore the entry node.
      if (equation != graph->getEntryNode()) {
        result.push_back(equation);
      }
    }

    return result;
  }

  /// Perform a reverse post-order visit of the dependency graph and get
  /// the ordered scalar equation descriptors.
  std::vector<ScalarEquationDescriptor> reversePostOrder() const {
    auto result = postOrder();
    std::reverse(result.begin(), result.end());
    return result;
  }

private:
  std::vector<ScalarEquationDescriptor>
  addEquationsAndMapWrites(llvm::ArrayRef<EquationProperty> equations) {
    std::vector<ScalarEquationDescriptor> vertices;

    // Differentiate the mutexes to enable more parallelism.
    std::mutex graphMutex;
    std::mutex writesMapMutex;

    auto mapFn = [&](const EquationProperty &equationProperty) {
      std::vector<Access> writeAccesses =
          ArrayEquationTraits::getWrites(&equationProperty);

      if (writeAccesses.empty()) {
        return;
      }

      const Access &writeAccess =
          getAccessWithProperty(writeAccesses, [](const Access &access) {
            return access.getAccessFunction().isAffine();
          });

      const AccessFunction &accessFunction = writeAccess.getAccessFunction();

      for (Point equationIndices :
           ArrayEquationTraits::getIterationRanges(&equationProperty)) {
        std::unique_lock<std::mutex> graphLockGuard(graphMutex);

        ScalarEquationDescriptor scalarEquationDescriptor =
            graph->addVertex(ScalarEquation(equationProperty, equationIndices));

        vertices.push_back(scalarEquationDescriptor);
        graphLockGuard.unlock();

        IndexSet writtenIndices(accessFunction.map(equationIndices));
        std::lock_guard<std::mutex> writesMapLockGuard(writesMapMutex);

        writesMap.emplace(writeAccess.getVariable(),
                          WriteInfo(*graph, writeAccess.getVariable(),
                                    scalarEquationDescriptor,
                                    std::move(writtenIndices)));
      }
    };

    mlir::parallelForEach(getContext(), equations, mapFn);
    return vertices;
  }

  void mapWrites() {
    // Differentiate the mutexes to enable more parallelism.
    std::mutex graphMutex;
    std::mutex writesMapMutex;

    auto mapFn = [&](ScalarEquationDescriptor equationDescriptor) {
      std::unique_lock<std::mutex> equationLockGuard(graphMutex);
      const ScalarEquation &scalarEquation = graph[equationDescriptor];
      equationLockGuard.unlock();

      const auto &vectorEquationProperty = scalarEquation.getProperty();

      std::vector<Access> writeAccesses =
          ArrayEquationTraits::getWrites(&vectorEquationProperty);

      if (!writeAccesses.empty()) {
        const Access &writeAccess =
            getAccessWithProperty(writeAccesses, [](const Access &access) {
              return access.getAccessFunction().isAffine();
            });

        const auto &accessFunction = writeAccess.getAccessFunction();

        IndexSet writtenIndices(accessFunction.map(scalarEquation.getIndex()));
        std::lock_guard<std::mutex> writesMapLockGuard(writesMapMutex);

        writesMap.emplace(writeAccess.getVariable(),
                          WriteInfo(*graph, writeAccess.getVariable(),
                                    equationDescriptor,
                                    std::move(writtenIndices)));
      }
    };

    mlir::parallelForEach(getContext(), graph->verticesBegin(),
                          graph->verticesEnd(), mapFn);
  }

  /// Determine the dependencies among the equations.
  template <typename EquationsIt>
  void addEdges(EquationsIt equationsBeginIt, EquationsIt equationsEndIt) {
    std::mutex graphMutex;

    auto mapFn = [&](ScalarEquationDescriptor equationDescriptor) {
      std::unique_lock<std::mutex> equationLockGuard(graphMutex);
      const ScalarEquation &scalarEquation = (*graph)[equationDescriptor];
      equationLockGuard.unlock();

      auto reads = ArrayEquationTraits::getReads(&scalarEquation.getProperty());

      for (const Access &read : reads) {
        auto readIndexes =
            read.getAccessFunction().map(scalarEquation.getIndex());

        auto writeInfos = writesMap.equal_range(read.getVariable());

        for (const auto &[variableId, writeInfo] :
             llvm::make_range(writeInfos.first, writeInfos.second)) {
          const auto &writtenIndexes = writeInfo.getWrittenVariableIndexes();

          if (writtenIndexes == readIndexes) {
            std::lock_guard<std::mutex> edgeLockGuard(graphMutex);
            graph->addEdge(equationDescriptor, writeInfo.getEquation());
          }
        }
      }
    };

    mlir::parallelForEach(getContext(), equationsBeginIt, equationsEndIt,
                          mapFn);
  }

  const Access &
  getAccessWithProperty(llvm::ArrayRef<Access> accesses,
                        std::function<bool(const Access &)> preferenceFn) {
    assert(!accesses.empty());
    auto it = llvm::find_if(accesses, preferenceFn);

    if (it == accesses.end()) {
      it = accesses.begin();
    }

    return *it;
  }
};
} // namespace marco::modeling

#endif // MARCO_MODELING_SCALAREQUATIONSDEPENDENCYGRAPH_H
